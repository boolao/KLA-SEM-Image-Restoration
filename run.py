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
# INFERENCE ENGINE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="KLA Inference Engine")
    parser.add_argument('input_dir', type=str, help="Path to input directory containing noisy .npy files")
    parser.add_argument('output_dir', type=str, help="Path to output directory to save restored .npy files")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

    # 1. Load the Model Safely
    model = SR_UNet_Residual().to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'sr_unet_true_ultimate.pth')
    
    if not os.path.exists(model_path):
        model_path = 'sr_unet_true_ultimate.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model weights at {model_path}.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Smart & Safe File Discovery (Handles nested folders but ignores Macs/Hidden)
    test_files = []
    actual_input_dir = args.input_dir
    
    for root, dirs, files in os.walk(args.input_dir):
        # Prevent searching inside hidden folders or __MACOSX
        dirs[:] = [d for d in dirs if not d.startswith('.') and '__MACOSX' not in d]
        
        valid_files = [f for f in files if f.endswith('.npy') and not f.startswith('.')]
        if valid_files:
            test_files = sorted(valid_files)
            actual_input_dir = root 
            break

    if not test_files:
        raise FileNotFoundError(f"No valid .npy files found in {args.input_dir} or its subdirectories.")

    # 3. Execute Inference with Dimension Safety Nets
    with torch.no_grad():
        for fname in tqdm(test_files, desc="Restoring Images"):
            lr_np = np.load(os.path.join(actual_input_dir, fname)).astype(np.float32)
            lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)

            # Safety Net 1: Force Normalization if input is 0-255 instead of 0-1
            if lr_tensor.max() > 2.0:
                lr_tensor = lr_tensor / 255.0

            # Safety Net 2: Dynamic Padding for odd-sized images (Divisible by 4 check)
            _, _, h, w = lr_tensor.shape
            pad_h = (4 - (h % 4)) % 4
            pad_w = (4 - (w % 4)) % 4
            
            if pad_h > 0 or pad_w > 0:
                lr_tensor = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            # Pass through the network
            pred_tensor = model(lr_tensor)
            
            # Safety Net 3: Crop back to exact target size (original size * 2)
            out_h, out_w = h * 2, w * 2
            pred_tensor = pred_tensor[:, :, :out_h, :out_w]
            
            # Safety Net 4: Catch NaN/Inf Math Errors
            pred_tensor = torch.nan_to_num(pred_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
            
            # Save strictly to the requested output directory
            pred_np = pred_tensor.squeeze().cpu().numpy()
            np.save(os.path.join(args.output_dir, fname), pred_np)

if __name__ == '__main__':
    main()
