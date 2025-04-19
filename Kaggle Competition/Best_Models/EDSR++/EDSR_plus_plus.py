# Full EDSR++ Patch-Based Super-Resolution Training with AMP and Charbonnier Loss
# Optimized for 2x RTX 3060 (12GB VRAM) with DataParallel

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler, autocast

# ----------------------------
# Charbonnier Loss
# ----------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

# ----------------------------
# Model: EDSR++
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x) * self.res_scale

class EDSRPP(nn.Module):
    def __init__(self, scale_factor=4, num_blocks=32, num_feats=256):
        super(EDSRPP, self).__init__()
        self.head = nn.Conv2d(3, num_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_feats) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_feats, num_feats * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_feats, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x

# ----------------------------
# Dataset with Patch Sampling and Augmentation
# ----------------------------
class PatchDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=48, stride=48):
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_paths)

    def extract_patches(self, img, patch_size, stride):
        _, h, w = img.shape
        patches = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = img[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        return patches

    def __getitem__(self, idx):
        lr = self.transform(Image.open(self.lr_paths[idx]).convert('RGB'))
        hr = self.transform(Image.open(self.hr_paths[idx]).convert('RGB'))

        lr_patches = self.extract_patches(lr, self.patch_size, self.stride)
        hr_patches = self.extract_patches(hr, self.patch_size * 4, self.stride * 4)

        rand_idx = torch.randint(0, len(lr_patches), (1,)).item()
        lr_patch, hr_patch = lr_patches[rand_idx], hr_patches[rand_idx]

        # Basic augmentation
        if torch.rand(1).item() > 0.5:
            lr_patch = torch.flip(lr_patch, dims=[2])
            hr_patch = torch.flip(hr_patch, dims=[2])

        return lr_patch, hr_patch

# ----------------------------
# Training Loop with AMP
# ----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSRPP().to(device)
    model = torch.nn.DataParallel(model)

    dataset = PatchDataset("../gnr638/train-kaggle/train-kaggle/lr", "../gnr638/train-kaggle/train-kaggle/hr")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    criterion = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()

            with autocast():
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/200, Loss: {epoch_loss / len(loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/edsrpp_epoch{epoch+1}.pth")

# ----------------------------
# Inference Script
# ----------------------------
def test(model_path, test_dir="../gnr638/lr/lr", output_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSRPP().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.ToTensor()

    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            lr_img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
            lr_tensor = transform(lr_img).unsqueeze(0).to(device)

            with autocast():
                sr_tensor = model(lr_tensor)

            save_image(sr_tensor, os.path.join(output_dir, fname))

if __name__ == "__main__":
    # Uncomment one of the below:
    train()
    test("checkpoints/edsrpp_epoch200.pth")
