import torch.nn as nn
import torch

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x + residual
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.down1 = ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = ConvBNReLU(256, 512, kernel_size=3, stride=2, padding=1)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(512)

    def forward(self, x):
        features = {}

        x = self.conv1(x)

        x = self.down1(x)
        x = self.res1(x)
        features['level1'] = x

        x = self.down2(x)
        x = self.res2(x)
        features['level2'] = x

        x = self.down3(x)
        x = self.res3(x)
        features['level3'] = x

        x = self.down4(x)
        x = self.res4(x)
        features['level4'] = x

        return features

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(512, 256)
        )
        self.conv4 = ResidualBlock(256)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(256, 128)
        )
        self.conv3 = ResidualBlock(128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64)
        )
        self.conv2 = ResidualBlock(64)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(64, 32)
        )
        self.conv1 = ResidualBlock(32)

        self.alpha_out = nn.Sequential(
            ConvBNReLU(32, 16),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, encoder_features):
        x = encoder_features['level4']

        x = self.up4(x)
        x = x + encoder_features['level3']
        x = self.conv4(x)

        x = self.up3(x)
        x = x + encoder_features['level2']
        x = self.conv3(x)

        x = self.up2(x)
        x = x + encoder_features['level1']
        x = self.conv2(x)

        x = self.up1(x)
        x = self.conv1(x)

        alpha = self.alpha_out(x)

        return alpha

class SemanticRefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=4)
        self.decoder = Decoder()

    def forward(self, rgb, base_alpha):
        x = torch.cat([rgb, base_alpha], dim=1)

        features = self.encoder(x)

        semantic_alpha = self.decoder(features)

        return semantic_alpha