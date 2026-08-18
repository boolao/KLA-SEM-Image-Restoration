import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# =====================================================================
# INTEGRATED MODEL ARCHITECTURE
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SR_UNet_Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(128, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(128, 64)
        
        self.sr_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.sr_conv = ConvBlock(32, 32)
        
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        base_upscale = F.interpolate(x, scale_factor=2.0, mode='bicubic', align_corners=False)
        
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        b = self.bottleneck(self.pool2(x2))
        
        u1 = self.up_conv1(torch.cat([x2, self.up1(b)], dim=1))
        u2 = self.up_conv2(torch.cat([x1, self.up2(u1)], dim=1))
        
        residual = self.out(self.sr_conv(self.sr_up(u2)))
        
        return base_upscale + residual
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="KLA Inference Engine")
    parser.add_argument('input_dir', type=str, help="Path to input directory containing noisy .npy files")
    parser.add_argument('output_dir', type=str, help="Path to output directory to save restored .npy files")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)

    model = SR_UNet_Residual().to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'sr_unet_true_ultimate.pth')
    
    if not os.path.exists(model_path):
        model_path = 'sr_unet_true_ultimate.pth'

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    test_files = []
    actual_input_dir = args.input_dir
    for root, dirs, files in os.walk(args.input_dir):
        if '__MACOSX' in root:
            continue
        valid_files = [f for f in files if f.endswith('.npy') and not f.startswith('._')]
        if valid_files:
            test_files = sorted(valid_files)
            actual_input_dir = root 
            break

    with torch.no_grad():
        for fname in tqdm(test_files, desc="Restoring Images"):
            lr_np = np.load(os.path.join(actual_input_dir, fname)).astype(np.float32)
            lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)

            pred_tensor = model(lr_tensor)
            
            pred_tensor = torch.nan_to_num(pred_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
            
            pred_np = pred_tensor.squeeze().cpu().numpy()
            
            np.save(os.path.join(args.output_dir, fname), pred_np)

if __name__ == '__main__':
    main()
