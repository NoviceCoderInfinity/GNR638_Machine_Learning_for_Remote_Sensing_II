import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ---- EDSR Model (baseline with scale=4x) ----
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class EDSR(nn.Module):
    def __init__(self, scale_factor=4, num_blocks=16, num_features=64):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(3, num_features, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)],
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.upsample(x)
        return x


# ---- Dataset Class ----
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]) if hr_dir else None

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_img = cv2.imread(self.lr_files[idx])
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lr_tensor = ToTensor()(lr_img)

        if self.hr_files:
            hr_img = cv2.imread(self.hr_files[idx])
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hr_tensor = ToTensor()(hr_img)
            return lr_tensor, hr_tensor
        else:
            return lr_tensor, os.path.basename(self.lr_files[idx])


# ---- Training Function ----
def train(model, loader, num_epochs=100, lr=1e-4, device='cuda'):
    model.train()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(num_epochs):
        total_loss = 0
        for lr_img, hr_img in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            sr = model(lr_img)
            loss = criterion(sr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.6f}")


# ---- Inference Function ----
def inference(model, loader, output_dir='outputs/', device='cuda'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for lr_img, filename in tqdm(loader, desc='Generating images'):
            lr_img = lr_img.to(device)
            sr = model(lr_img).clamp(0, 1)
            sr_img = sr[0].cpu().permute(1, 2, 0).numpy() * 255
            sr_img = cv2.cvtColor(sr_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, filename[0]), sr_img)


# ---- Main Script ----
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EDSR(scale_factor=4)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    model = model.to(device)

    # --- Training ---
    train_data = SRDataset('../gnr638/train-kaggle/train-kaggle/lr', '../gnr638/train-kaggle/train-kaggle/hr')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    train(model, train_loader, num_epochs=50, device=device)

    torch.save(model.state_dict(), 'edsr_4x_rgb.pth')

    # --- Inference ---
    test_data = SRDataset('../gnr638/lr/lr')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    inference(model, test_loader, output_dir='outputs/', device=device)
