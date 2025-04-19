import torch
import torch.nn as nn

class RCAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        return res + x

class ResidualGroup(nn.Module):
    def __init__(self, channels, num_rcab):
        super().__init__()
        modules = [RCAB(channels) for _ in range(num_rcab)]
        modules.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        return res + x

class RCAN(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, channels=64, num_rg=10, num_rcab=20, scale=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_ch, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualGroup(channels, num_rcab) for _ in range(num_rg)],
                                  nn.Conv2d(channels, channels, 3, 1, 1))
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        self.tail = nn.Conv2d(channels, out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x) + x
        x = self.upsample(res)
        x = self.tail(x)
        return x

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.cuda.amp import autocast, GradScaler
from torch import optim
from tqdm import tqdm

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=64):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.filenames = os.listdir(lr_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.filenames[idx])

        lr_img = cv2.imread(lr_path).astype(np.float32) / 255.0
        hr_img = cv2.imread(hr_path).astype(np.float32) / 255.0

        if self.patch_size > 0:
            H, W, _ = lr_img.shape
            x = np.random.randint(0, H - self.patch_size)
            y = np.random.randint(0, W - self.patch_size)
            lr_img = lr_img[x:x+self.patch_size, y:y+self.patch_size]
            hr_img = hr_img[x*4:(x+self.patch_size)*4, y*4:(y+self.patch_size)*4]

        lr_img = ToTensor()(lr_img)
        hr_img = ToTensor()(hr_img)

        return lr_img, hr_img

# Paths and settings
lr_dir = "../gnr638/train-kaggle/train-kaggle/lr"
hr_dir = "../gnr638/train-kaggle/train-kaggle/hr"
batch_size = 8
num_epochs = 50
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader
train_dataset = SRDataset(lr_dir, hr_dir, patch_size=64)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Model, Loss, Optimizer
model = RCAN(scale=4).to(device)
model = torch.nn.DataParallel(model)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = GradScaler()

# Training
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        optimizer.zero_grad()
        with autocast():
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} - Avg Loss: {epoch_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "rcan_4xSR.pth")
