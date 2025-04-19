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


# ===== Load Model and Run Inference =====
def run_inference(checkpoint_path, test_lr_path, output_path):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Initialize model
    model = ViTSR().to(device)

    # Load checkpoint
    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Dataset and loader
    test_dataset = SRDataset(test_lr_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Run inference
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for idx, lr in enumerate(tqdm(test_loader, desc="Generating HR images")):
            lr = lr.to(device)
            sr = model(lr)
            sr = F.interpolate(sr, size=(1080, 1920), mode='bicubic')
            vutils.save_image(sr.clamp(0, 1), os.path.join(output_path, f"sr_{idx:03d}.png"))

# ===== Main =====
if __name__ == '__main__':
    checkpoint = 'checkpoint_epoch_20.pth'  # Change to your desired checkpoint
    test_lr = '../gnr638/lr/lr'  # Make sure this is the correct test path now
    save_results_to = 'results'

    run_inference(checkpoint, test_lr, save_results_to)
