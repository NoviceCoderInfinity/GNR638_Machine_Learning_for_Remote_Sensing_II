import os
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RCAN(scale=4).to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("rcan_4xSR.pth"))
model.eval()

test_lr_dir = "../gnr638/lr/lr"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(test_lr_dir)):
    lr_path = os.path.join(test_lr_dir, filename)
    output_path = os.path.join(output_dir, filename)

    lr_img = cv2.imread(lr_path).astype(np.float32) / 255.0
    lr_tensor = ToTensor()(lr_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0, 1)

    sr_img = sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    sr_img = (sr_img * 255.0).astype(np.uint8)
    cv2.imwrite(output_path, sr_img)

print(f"Generated {len(os.listdir(test_lr_dir))} SR images in '{output_dir}'")
