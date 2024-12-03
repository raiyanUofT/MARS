import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. RES-SE Block Definition
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

# 2. Radar-ViT with Multi-Head Attention
class RadarViT(nn.Module):
    def __init__(self, input_dim=5, embed_dim=16, num_heads=4, mlp_ratio=4.0):
        super(RadarViT, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        B, C, N, F = x.size()

        x = x.view(B, N, F)
        x = self.proj(x)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x.view(B, C, N, -1)

# 3. LHVIT Multi-Head Model
class LHVITMultiHead(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LHVITMultiHead, self).__init__()
        self.feature_extractor = nn.Sequential(
            ResSEBlock(in_channels),
            nn.MaxPool2d((2, 1)),
            ResSEBlock(in_channels),
            nn.MaxPool2d((2, 1))
        )
        self.feature_enhancer = RadarViT(input_dim=5, embed_dim=16, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # 30% chance to drop neurons
            nn.Linear(256, num_classes)  # Adjust based on flattened input size
        )


    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)

        # Feature enhancement with RadarViT
        x = self.feature_enhancer(x)

        # Flatten directly without pooling
        x = x.view(x.size(0), -1)  # Flatten to [B, 256]

        # Pass through the classifier
        x = self.classifier(x)  # Shape: [B, num_classes]

        return x