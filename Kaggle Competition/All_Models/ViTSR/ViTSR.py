import os
import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from einops import rearrange


# ===== ViT Super-Resolution Model =====
class ViTSR(nn.Module):
    def __init__(self, upscale=4, dim=512, depth=6, heads=8, mlp_dim=1024, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.upscale = upscale

        self.input_proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        num_patches = (480 // patch_size) * (270 // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_proj = nn.Linear(dim, 3 * self.upscale * self.upscale)


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.input_proj(x)  # (B, dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b c h w -> b (h w) c')  # flatten to patch tokens

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, dim)
        x = x + self.pos_embed[:, :x.size(1), :]

        x = self.transformer(x)
        x = x[:, 1:]  # remove cls token
        x = self.output_proj(x)  # (B, num_patches, 3 * upscale^2)

        # Reshape back to image
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        assert grid_h * grid_w == x.size(1), f"Patch count mismatch: {grid_h * grid_w} != {x.size(1)}"

        x = rearrange(
            x, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
            h=grid_h, w=grid_w, p1=self.upscale, p2=self.upscale, c=3
        )
        return x




# ===== Dataset Loader =====
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None, transform=None):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png'))) if hr_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_paths[idx]).convert("RGB")
        lr = self.transform(lr)
        if self.hr_paths:
            hr = Image.open(self.hr_paths[idx]).convert("RGB")
            hr = self.transform(hr)
            return lr, hr
        return lr


# ===== Training =====
def train(model, train_loader, optimizer, criterion, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            sr = F.interpolate(sr, size=(1080, 1920), mode='bicubic')  # match resolution
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")
        torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


# ===== Inference =====
def inference(model, test_loader, save_dir, device):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, lr in enumerate(tqdm(test_loader, desc="Generating HR images")):
            lr = lr.to(device)
            sr = model(lr)
            sr = F.interpolate(sr, size=(1080, 1920), mode='bicubic')
            vutils.save_image(sr.clamp(0, 1), os.path.join(save_dir, f"sr_{idx:03d}.png"))
            


# ===== Main Pipeline =====
def main():
    # Setup
    batch_size = 8
    num_epochs = 50
    lr_path = '../gnr638/train-kaggle/train-kaggle/lr'
    hr_path = '../gnr638/train-kaggle/train-kaggle/hr'
    test_lr_path = '../gnr638/lr/lr'
    output_path = 'results'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ViTSR()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Training
    train_dataset = SRDataset(lr_path, hr_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train(model, train_loader, optimizer, criterion, device, num_epochs=num_epochs)

    # Inference
    test_dataset = SRDataset(test_lr_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    inference(model, test_loader, output_path, device)


if __name__ == '__main__':
    main()
