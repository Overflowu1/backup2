import torch
import torch.nn as nn


class MultiscaleSparseMARNet(nn.Module):
    """轻量级金属伪影校正网络"""

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 多尺度伪影提取
        self.artifact_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels, 1),
                nn.ReLU()
            ) for _ in range(3)  # 三种尺度
        ])

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 5, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, metal_mask):
        # 金属感知输入
        x = torch.cat([x, metal_mask], dim=1)

        # 特征提取
        feats = self.encoder(x)

        # 多尺度伪影提取
        artifact_feats = []
        for branch in self.artifact_branches:
            artifact_feats.append(branch(feats))

        # 上采样并拼接特征
        artifact_feats[1] = F.interpolate(artifact_feats[1], scale_factor=2)
        artifact_feats[2] = F.interpolate(artifact_feats[2], scale_factor=4)
        combined = torch.cat(artifact_feats, dim=1)

        # 残差学习
        residual = self.decoder(combined)
        return x[:, :1] - residual  # 伪影校正