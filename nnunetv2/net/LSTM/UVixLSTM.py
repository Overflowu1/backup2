import torch
import torch.nn as nn
from einops import rearrange
from monai.networks.blocks import PatchEmbeddingBlock
import einops
import torch.nn.functional as F
from nnunetv2.net.LSTM.VisionLSTM import *

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm3d(width)

        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm3d(width)

        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels,
                 depth=8,  # Reduce depth
                 dim=128,  # Reduce dimension
                 drop_path_rate=0.0,
                 stride=None,
                 alternation="bidirectional",
                 drop_path_decay=False,
                 legacy_norm=False):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        patch_size = (2, 2, 2)
        img_size = (10, 6, 10)

        self.patch_embed = PatchEmbeddingBlock(
            in_channels=out_channels * 8,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=dim,
            num_heads=1,
            proj_type='conv',  # Update to the new argument name
            spatial_dims=3
        )

        self.conv2 = nn.Conv3d(out_channels * 8, dim,  # Ensure channels match the dimension
                               kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(dim)
        self.alternation = alternation
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        directions = []
        if alternation == "bidirectional":
            for i in range(depth):
                if i % 2 == 0:
                    directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
                else:
                    directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        else:
            raise NotImplementedError(f"invalid alternation '{alternation}'")

        self.blocks = nn.ModuleList(
            [
                ViLBlock(
                    dim=dim,
                    drop_path=dpr[i],
                    direction=directions[i],
                )
                for i in range(depth)
            ]
        )
        if legacy_norm:
            self.legacy_norm = LayerNorm(dim, bias=False)
        else:
            self.legacy_norm = nn.Identity()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2], dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        print(f"Size before patch embedding: {x.size()}")
        x = self.patch_embed(x)
        print(f"Size after patch embedding: {x.size()}")
        x = einops.rearrange(x, "b ... d -> b (...) d")

        for block in self.blocks:
            x = block(x)
        x = self.legacy_norm(x)
        x = self.norm(x)

        x = rearrange(x, "b (x y z) c -> b c x y z", x=self.output_shape[0], y=self.output_shape[1], z=self.output_shape[2])
        return x, x1, x2, x3



class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=scale_factor * 2, mode='trilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        if x.shape[2] == 3:
            x = self.upsample1(x)
        else:
            x = self.upsample(x)
        if x_concat is not None:
            if x.shape != x_concat.shape:
                diff = [x_concat.size(i) - x.size(i) for i in range(2, 5)]
                padding = [diff[2] // 2, diff[2] - diff[2] // 2,
                           diff[1] // 2, diff[1] - diff[1] // 2,
                           diff[0] // 2, diff[0] - diff[0] // 2]
                x = F.pad(x, padding)
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x



class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv3d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)

        return x


class UVixLSTM(nn.Module):
    def __init__(self, class_num, img_dim=96,
                 in_channels=1,
                 out_channels=64,
                 depth=8,  # Reduce depth
                 dim=128):  # Reduce dimension
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               depth, dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        print(x.size(), x1.size(), x2.size(), x3.size())
        x = self.decoder(x, x1, x2, x3)

        return x


if __name__ == '__main__':
    model = UVixLSTM(4)
    x = torch.randn(2, 1, 160, 96, 160)
    y = model(x)
    print(y.size())
