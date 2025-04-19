import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

# -------------------------------
# Dataset
# -------------------------------
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=96, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = Image.open(os.path.join(self.lr_dir, self.lr_images[idx])).convert('RGB')
        hr = Image.open(os.path.join(self.hr_dir, self.hr_images[idx])).convert('RGB')

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        # Random crop for training
        _, H, W = lr.shape
        ps = self.patch_size
        x = torch.randint(0, W - ps, (1,))
        y = torch.randint(0, H - ps, (1,))
        lr_crop = lr[:, y:y+ps, x:x+ps]
        hr_crop = hr[:, y*4:y*4+ps*4, x*4:x*4+ps*4]

        return lr_crop, hr_crop

# -------------------------------
# Perceptual Loss
# -------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:16])  # up to relu_2_2
        self.loss = nn.L1Loss()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        return self.loss(self.features(sr), self.features(hr))


# -------------------------------
# NLSA Model
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class NonLocalSparseAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 1)
        self.up = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        res = x
        x = self.down(x)
        q = rearrange(x, 'b c h w -> b (h w) c')
        k = rearrange(x, 'b c h w -> b c (h w)')
        attn = torch.bmm(q, k)
        attn = self.softmax(attn / (C ** 0.5))

        v = rearrange(x, 'b c h w -> b c (h w)')
        out = torch.bmm(attn, v.transpose(1, 2))
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        out = self.up(out)
        return res + out

class UpsampleBlock(nn.Module):
    def __init__(self, scale, channels):
        super().__init__()
        layers = []
        for _ in range(int(scale).bit_length() - 1):
            layers += [
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(True)
            ]
        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample(x)

class NLSA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=16, upscale=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks // 2)],
            NonLocalSparseAttention(channels),
            *[ResidualBlock(channels) for _ in range(num_blocks // 2)],
        )
        self.upsample = UpsampleBlock(upscale, channels)
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.upsample(res + x)
        return self.tail(x)

# -------------------------------
# Training + Inference
# -------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.ToTensor()

    dataset = SRDataset("../gnr638/train-kaggle/train-kaggle/lr", "../gnr638/train-kaggle/train-kaggle/hr", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    model = NLSA().to(device)
    model = nn.DataParallel(model)

    l1_loss = nn.L1Loss()
    vgg_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for epoch in range(100):
        model.train()
        running_loss = 0
        for lr, hr in tqdm(loader, desc=f"Epoch {epoch+1}"):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            with autocast():
                sr = model(lr)
                loss = 0.8 * l1_loss(sr, hr) + 0.2 * vgg_loss(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {running_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"nlsa_epoch{epoch+1}.pth")

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.ToTensor()

    model = NLSA().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("nlsa_epoch100.pth", map_location=device))
    model.eval()

    test_dir = "../gnr638/lr/lr"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(test_dir)), desc="Testing"):
            img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
            lr = transform(img).unsqueeze(0).to(device)

            # Use AMP and half precision to save VRAM
            with autocast():
                sr = model(lr.half())  # Move LR to float16
                sr = sr.float()        # Convert back to float32 for saving

            save_image(sr, os.path.join(output_dir, fname))
            torch.cuda.empty_cache()  # Free memory after each image

import numpy as np
def test_model_tiled(tile_size=128, tile_overlap=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.ToTensor()

    model = NLSA().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("nlsa_epoch100.pth", map_location=device))
    model.eval()

    test_dir = "../gnr638/lr/lr"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    scale = 4  # 4x super-resolution

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(test_dir)), desc="Tiled Testing"):
            img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
            lr = transform(img).unsqueeze(0).to(device)

            b, c, h, w = lr.shape
            tile = tile_size
            overlap = tile_overlap
            stride = tile - overlap

            out = torch.zeros((b, c, h * scale, w * scale), device=device)
            count = torch.zeros((b, 1, h * scale, w * scale), device=device)

            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y0, x0 = y, x
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)

                    lr_tile = lr[:, :, y0:y1, x0:x1]

                    # Padding if necessary
                    pad_h = tile - lr_tile.shape[2]
                    pad_w = tile - lr_tile.shape[3]
                    if pad_h > 0 or pad_w > 0:
                        lr_tile = torch.nn.functional.pad(lr_tile, (0, pad_w, 0, pad_h), mode="reflect")

                    # Run in float32 to avoid conv half/float mismatch
                    sr_tile = model(lr_tile.float())

                    # Remove padding from SR tile
                    sr_tile = sr_tile[:, :, : (y1 - y0) * scale, : (x1 - x0) * scale]

                    out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += sr_tile
                    count[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += 1

                    torch.cuda.empty_cache()

            sr = out / count
            save_image(sr.clamp(0, 1), os.path.join(output_dir, fname))


# -------------------------------
# Run Script
# -------------------------------
if __name__ == "__main__":
    # train_model()
    test_model_tiled(tile_size=128, tile_overlap=32)