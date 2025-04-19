import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# --------- SRCNN for RGB ---------
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)

# --------- Dataset for RGB SR ---------
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)]) if hr_dir else None

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_img = cv2.imread(self.lr_files[idx])  # BGR image
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        lr_img = lr_img.astype(np.float32) / 255.0
        lr_up = cv2.resize(lr_img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        lr_tensor = ToTensor()(lr_up)

        if self.hr_files:
            hr_img = cv2.imread(self.hr_files[idx])
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = hr_img.astype(np.float32) / 255.0
            hr_tensor = ToTensor()(hr_img)
            return lr_tensor, hr_tensor
        else:
            return lr_tensor, os.path.basename(self.lr_files[idx])

# --------- Training Function ---------
def train_model(model, train_loader, num_epochs=100, lr=1e-4, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader):.6f}")

# --------- Inference Function ---------
def super_resolve(model, test_loader, output_dir, device='cuda'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for lr_imgs, filenames in tqdm(test_loader, desc="Generating SR RGB images"):
            lr_imgs = lr_imgs.to(device)
            outputs = model(lr_imgs).clamp(0.0, 1.0)

            for i in range(outputs.size(0)):
                output = outputs[i].cpu().permute(1, 2, 0).numpy() * 255.0
                output = output.astype(np.uint8)
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, filenames[i]), output_bgr)

# --------- Main ---------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use both GPUs if available
    model = SRCNN()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Load training dataset
    train_dataset = SRDataset('../gnr638/train-kaggle/train-kaggle/lr', '../gnr638/train-kaggle/train-kaggle/hr')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    print("Training RGB SRCNN...")
    train_model(model, train_loader, num_epochs=100, device=device)

    torch.save(model.state_dict(), 'srcnn_rgb_4x.pth')

    # Generate SR RGB images
    test_dataset = SRDataset('../gnr638/lr/lr')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    print("Generating RGB SR test images...")
    super_resolve(model, test_loader, output_dir='outputs_rgb/', device=device)
