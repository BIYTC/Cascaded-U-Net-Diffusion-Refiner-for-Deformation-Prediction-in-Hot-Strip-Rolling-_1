import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusionUtils import ModelConfig


# Modified: 增强版时间嵌入
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = (dim + 1) // 2  # Modified: 修复维度计算问题
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1 + 1e-6)  # Added: 防止除零
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

        # Added: 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, t):
        t = t.float() * self.scale  # Modified: 增强时间嵌入表达
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# Added: 自适应归一化层
class AdaGN(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )

        # 动态计算分组数
        if dim >= 8:
            # 找到最大的可整除分组数（<=8）
            self.num_groups = 8
            while dim % self.num_groups != 0:
                self.num_groups -= 1
        else:
            self.num_groups = 1

        self.norm = nn.GroupNorm(self.num_groups, dim)
        # print(f"Created GroupNorm with {self.num_groups} groups for {dim} channels")  # 调试信息

    def forward(self, x, t_emb):
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)  # 使用嵌入后的时间参数
        x = self.norm(x)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


# Modified: 增强残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.config = config

        # Modified: 自适应归一化选择
        if config.use_adaGN:
            self.norm1 = AdaGN(in_channels, time_dim)
            self.norm2 = AdaGN(out_channels, time_dim)
        else:
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_embed = nn.Linear(time_dim, out_channels * 2)

        # Added: 通道注意力
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        ) if out_channels >= 8 else None

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(config.dropout)

        # Added: 跳跃连接自适应
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.norm1(x, t_emb) if self.config.use_adaGN else self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Modified: 增强时间嵌入融合
        # t_emb = t_emb[:, :, None, None]
        # scale, shift = t_emb.chunk(2, dim=1)
        # h = h * (1 + scale) + shift

        h = self.norm2(h, t_emb) if self.config.use_adaGN else self.norm2(h)
        h = self.conv2(F.silu(h))

        # Added: 通道注意力
        if self.attn is not None:
            attn = self.attn(h)
            h = h * attn

        return self.shortcut(x) + self.dropout(h)


# Modified: 改进的下采样块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels,
                          out_channels, time_dim, config)
            for i in range(config.num_res_blocks)
        ])
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)  # Modified: 替换原下采样

        # Added: 注意力层
        if len(self.blocks) in config.attn_layers:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=4)
        else:
            self.attn = None

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        if self.attn is not None:
            B, C, H, W = x.shape
            x_attn = x.view(B, C, -1).permute(2, 0, 1)
            x_attn = self.attn(x_attn, x_attn, x_attn)[0]
            x = x_attn.permute(1, 2, 0).view(B, C, H, W)
        return self.down(x)


# Modified: 改进的上采样块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, config):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Modified: 替换转置卷积
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(out_channels * 2 if i == 0 else out_channels,
                          out_channels, time_dim, config)
            for i in range(config.num_res_blocks)
        ])

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, t)
        return x


# Modified: 增强特征上采样（关键修改点）
class FeatureUpSample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(21, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8))
        )

        # 分解Sequential为可管理模块
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(128, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 16*16
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 32*32
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 64*64
            ResidualBlock(256, 128, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),        # 128*128
            ResidualBlock(128, 64, config.time_dim, config),
            nn.Conv2d(64, config.channel_mult[0], 3, padding=1)
        ])

    def forward(self, x, t):  # 新增时间参数
        x = self.fc(x)
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# Modified: 改进的Xt上采样（关键修改点）
class XtUpSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 分解encoder为独立模块
        self.encoder_blocks = nn.ModuleList([
            nn.Conv2d(1, 64, 3, padding=1),
            ResidualBlock(64, 128, config.time_dim, config),
            # nn.AvgPool2d(2),
            ResidualBlock(128, 256, config.time_dim, config),
            # nn.AvgPool2d(2),
        ])

        # 分解decoder为独立模块
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(256, 256, config.time_dim, config),   # 21*51
            nn.Upsample(scale_factor=(2,1), mode='bilinear'),   # 42*51
            ResidualBlock(256, 256, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),       # 84*102
            ResidualBlock(256, 128, config.time_dim, config),
            nn.Upsample(scale_factor=2, mode='bilinear'),       # 168*204
            ResidualBlock(128, 64, config.time_dim, config),
            nn.Upsample(size=(128, 128), mode='bilinear'),        # 强制对齐尺寸
            nn.Conv2d(64, config.channel_mult[0], 3, padding=1)
        ])

    def forward(self, x, t_emb):  # 新增时间参数
        # 处理encoder
        for module in self.encoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
            else:
                x = module(x)

        # 处理decoder
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
        return x


# Modified: 重构DXYHead（关键修改点）
class DXYHead(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config

        # 特征路径
        self.feature_upsample = FeatureUpSample(config)
        self.xt_upsample = XtUpSampler(config)

        # 初始融合
        self.init_conv = nn.Conv2d(config.channel_mult[0] * 2, config.channel_mult[0], 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = [config.channel_mult[i] for i in range(len(config.channel_mult))]
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], config.time_dim, config)
            )

        # 中间层
        self.mid_block = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], config.time_dim, config),
            ResidualBlock(channels[-1], channels[-1], config.time_dim, config)
        )

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                UpBlock(channels[i + 1], channels[i], config.time_dim, config)
            )

        # 最终投影
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], 32, 3, padding=1),   # 128*128
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Upsample(size=(21, 51)),
            nn.Conv2d(32, 1, 1),
            # nn.Tanh()  #  不能要tanh啊啊啊
        )
        #  从128*128到16*32
        # self.final = nn.Sequential(
        #     # 第1步：缩小高度和宽度（128x128 → 64x64）
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=1,
        #         kernel_size=(4, 4),
        #         stride=(2, 2),
        #         padding=(1, 1)
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1),
        #     # 第2步：仅缩小高度（64x64 → 32x64）
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=1,
        #         kernel_size=(4, 1),  # 仅在高度方向操作
        #         stride=(2, 1),
        #         padding=(1, 0)
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1),
        #     # 第3步：继续缩小高度（32x64 → 16x64）
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=1,
        #         kernel_size=(4, 1),
        #         stride=(2, 1),
        #         padding=(1, 0)
        #     ),
        #     nn.BatchNorm2d(1),
        #     # 第4步：缩小宽度（16x64 → 16x32）
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=1,
        #         kernel_size=(1, 4),  # 仅在宽度方向操作
        #         stride=(1, 2),
        #         padding=(0, 1)
        #     ),
        #     # nn.Tanh()
        #     )

        # 时间嵌入
        self.time_embed = TimeEmbedding(config.time_dim)

    def forward(self, backbone_feat, x_t, t):
        t = t.float()  # 新增类型转换
        # 特征处理（传入时间参数）
        feat = self.feature_upsample(backbone_feat, self.time_embed(t))  # 关键修改
        xt = self.xt_upsample(x_t, self.time_embed(t))  # 关键修改

        # 初始融合
        x = torch.cat([feat, xt], dim=1)
        x = self.init_conv(x)

        # 时间嵌入
        t_emb = self.time_embed(t)

        # 下采样
        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x, t_emb)

        # 中间处理
        x = self.mid_block[0](x, t_emb)
        x = self.mid_block[1](x, t_emb)

        # 上采样
        for block in self.up_blocks:
            x = block(x, skips.pop(), t_emb)

        return self.final(x).squeeze(1)


# Modified: 最终模型整合
class DiffusionModel(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.dx_head = DXYHead(config)
        self.dy_head = DXYHead(config)

    def forward(self, x_t, features, t):
        dx = self.dx_head(features, x_t[:, 0:1], t)
        dy = self.dy_head(features, x_t[:, 1:2], t)
        return torch.stack([dx, dy], dim=1)