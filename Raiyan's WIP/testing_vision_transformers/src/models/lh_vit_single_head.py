import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 1. RES-SE Block Definition (Lightweight Convolution + SE Attention)
class ResSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ResSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.se = SqueezeExcitation(in_channels, reduction)
        self.bn = nn.BatchNorm2d(in_channels)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.hardswish(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + identity

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction):
        super(SqueezeExcitation, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        out = self.pool(x).view(batch_size, channels)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out)).view(batch_size, channels, 1, 1)
        return x * out

# 2. RadarViT: Vision Transformer with Single Head Attention
class RadarViT(nn.Module):
    def __init__(self, dim=5, num_heads=1, mlp_ratio=4.0):
        super(RadarViT, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        B, C, N, F = x.size()
        x = x.view(B, N, F)  # Reshape to [B, N, F]

        # Apply single head attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out  # Residual connection

        # Apply MLP block with residual connection
        x = x + self.mlp(self.norm2(x))

        x = x.view(B, C, N, -1)  # Reshape back to [B, C, N, F]
        return x

# 3. LH-ViT Network: Combining Feature Extraction and Enhancement
class LHVITSingleHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):  # Adjust num_classes if needed
        super(LHVITSingleHead, self).__init__()
        self.feature_extractor = nn.Sequential(
            ResSEBlock(in_channels),
            nn.MaxPool2d((2, 1)),  # Pool along points, not features
            ResSEBlock(in_channels),
            nn.MaxPool2d((2, 1))
        )
        self.feature_enhancer = RadarViT(dim=5, num_heads=1)  # 5 features per point
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool to [B, C, 1, 1]
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        # Input: [B, 1, 64, 5]
        x = self.feature_extractor(x)  # Shape: [B, 1, 16, 5]
        x = self.feature_enhancer(x)   # Shape: [B, 1, 16, 5]
        x = self.classifier(x)         # Shape: [B, num_classes]
        return x
