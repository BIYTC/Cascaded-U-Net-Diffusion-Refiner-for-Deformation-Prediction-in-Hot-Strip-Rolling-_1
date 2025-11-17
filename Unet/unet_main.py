import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
from dataset import *


# 配置类示例
class Unet_ModelConfig:
    def __init__(self,
                 model_size='base',
                 input_dim=21,
                 output_channels=2):
  # 调整后的输入形状
        self.input_dim = input_dim
        self.output_channels = output_channels
        # 全连接层配置
        self.fc_layers = {
            'tiny': {
                'hidden_dims': [128, 256],
                'output_shape': (32, 32)
            },
            'base': {
                'hidden_dims': [256, 512, 1024],
                'output_shape': (64, 64)
            }
        }[model_size]

        # 调整后的UNet通道配置
        self.encoder_channels = {
            'tiny': [4, 8, 16],
            'base': [4, 8, 16, 32]
        }[model_size]

        self.decoder_channels = {
            'tiny': [8, 4],
            'base': [16, 8, 4]
        }[model_size]

        # 其他参数保持不变
        self.activation = nn.ReLU
        self.norm_layer = nn.BatchNorm2d
        self.dropout = 0.2
        self.pooling = nn.MaxPool2d(2)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

# UNet 模型定义
class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_transform = self._build_fc_transform(config)  # 新增特征变换模块
        self.encoder = self._build_encoder(config)
        self.bottleneck = self._build_bottleneck(config)
        self.decoder = self._build_decoder(config)
        self.final_conv = nn.Sequential(
            nn.Conv2d(config.decoder_channels[-1], 256, 3, padding=1),
            ResidualBlock(256),
            nn.Conv2d(256, 128, 1),
            nn.Upsample(scale_factor=1.5, mode='bilinear'),  # 中间放大
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((21, 51)),
            nn.Conv2d(128, config.output_channels, 3, padding=1)
        )

    def _build_fc_transform(self, config):
        layers = []
        input_dim = config.input_dim
        hidden_dims = config.fc_layers['hidden_dims']

        # 构建全连接层
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                config.activation(),
                nn.Dropout(config.dropout)
            ])
            input_dim = h_dim

        # 最终投影到目标形状
        target_elements = config.encoder_channels[0] * \
                          config.fc_layers['output_shape'][0] * \
                          config.fc_layers['output_shape'][1]

        layers.append(nn.Linear(input_dim, target_elements))
        return nn.Sequential(*layers)

    def _build_encoder(self, config):
        layers = []
        in_ch = config.encoder_channels[0]  # 注意此处起始通道已改变
        spatial_size = config.fc_layers['output_shape']

        # 验证输入尺寸合理性
        min_size = 2 ** (len(config.encoder_channels) - 1)
        assert spatial_size[0] >= min_size and spatial_size[1] >= min_size, \
            f"输入尺寸{spatial_size}不足以支持{len(config.encoder_channels)}层下采样"

        for out_ch in config.encoder_channels[1:]:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                config.norm_layer(out_ch),
                config.activation(),
                nn.Dropout(config.dropout),
                config.pooling  # 直接使用实例
            ))
            in_ch = out_ch
        return nn.ModuleList(layers)

    def _build_bottleneck(self, config):
        return nn.Sequential(
            nn.Conv2d(config.encoder_channels[-1], config.encoder_channels[-1], 3, padding=1),
            config.norm_layer(config.encoder_channels[-1]),
            config.activation(),
            nn.Conv2d(config.encoder_channels[-1], config.encoder_channels[-1], 3, padding=1),
            config.norm_layer(config.encoder_channels[-1]),
            config.activation()
        )

    def _build_decoder(self, config):
        layers = []
        for i in range(len(config.decoder_channels)):
            up_conv = nn.ConvTranspose2d(
                config.encoder_channels[-i-1],
                config.decoder_channels[i],
                kernel_size=2,
                stride=2
            )
            conv_block = nn.Sequential(
                nn.Conv2d(
                    config.decoder_channels[i] + config.encoder_channels[-i-2],
                    config.decoder_channels[i],
                    kernel_size=3,
                    padding=1
                ),
                config.norm_layer(config.decoder_channels[i]),
                config.activation(),
                nn.Dropout(config.dropout)
            )
            layers.append(nn.ModuleDict({'up_conv': up_conv, 'conv_block': conv_block}))
        return nn.ModuleList(layers)

    def forward(self, x):
        # 特征变换
        batch_size = x.size(0)
        x = self.fc_transform(x).view(
            batch_size,
            self.config.encoder_channels[0],
            *self.config.fc_layers['output_shape']
        )

        # 编码器
        skip_connections = [x]
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器
        for i, dec_layer in enumerate(self.decoder):
            x = dec_layer['up_conv'](x)
            skip = skip_connections[-(i + 2)]
            # 尺寸对齐
            _, _, h, w = skip.shape
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_layer['conv_block'](x)

        return self.final_conv(x)


# 训练框架
class UNetTrainer:
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config

        # 设备配置
        self.device = torch.device(config['device'])
        self.model = self._init_model().to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.75,
            patience=5
        )

        # 损失函数
        self.loss_fn = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'smooth_l1': nn.SmoothL1Loss(),
            'huber': nn.HuberLoss()
        }[config['loss_type']]

        # 日志记录
        self.writer = SummaryWriter(log_dir=config['log_dir'])

    def _init_model(self):
        model = UNet(self.model_config)
        if torch.cuda.device_count() > 1 and self.config['multi_gpu']:
            model = nn.DataParallel(model)
        return model

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        with tqdm(train_loader, desc="Training") as pbar:
            for features, targets, *_ in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        return avg_loss, epoch_time

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for features, targets, *_ in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def test(self, test_loader):
        self.model.eval()
        # 20250322 新增部分：初始化结果存储结构
        result_dict = {
            'preds': [],
            'inputs': [],
            'targets': [],
            'max_dx': [],
            'min_dx': [],
            'max_dy': [],
            'min_dy': [],
            'sample_indices': []  # 记录样本在数据集中的原始索引
        }
        with torch.no_grad():
            for i, (features, target, max_dx, min_dx, max_dy, min_dy) in tqdm(enumerate(test_loader)):
                if i % 10 == 0:
                    print("\nDoing batch {}\n".format(i))
                    st = time.time()
                    features = features.to(self.device).float()
                    target = target.to(self.device).float()
                    b = target.shape[0]
                    outputs = self.model(features)
                    # 新增部分：收集结果
                    batch_indices = [i * test_loader.batch_size + j for j in range(b)]  # 计算样本全局索引
                    result_dict['preds'].append(outputs.cpu().detach())  # 存储为CPU张量
                    result_dict['inputs'].append(features.cpu().detach())
                    result_dict['targets'].append(target.cpu().detach())
                    result_dict['max_dx'].append(max_dx.cpu().detach())
                    result_dict['min_dx'].append(min_dx.cpu().detach())
                    result_dict['max_dy'].append(max_dy.cpu().detach())
                    result_dict['min_dy'].append(min_dy.cpu().detach())
                    result_dict['sample_indices'].extend(batch_indices)
                    print("Time taken: ", time.time() - st)
        # 新增部分：保存结果到文件
        save_path = "test_results.pth"
        torch.save({
            'preds': torch.cat(result_dict['preds'], dim=0),  # 合并为完整张量 [N,2,21,51]
            'inputs': torch.cat(result_dict['inputs'], dim=0),  # [N,...]
            'targets': torch.cat(result_dict['targets'], dim=0),  # [N,...]
            'max_dx': torch.cat(result_dict['max_dx'], dim=0),  # [N]
            'min_dx': torch.cat(result_dict['min_dx'], dim=0),  # [N]
            'max_dy': torch.cat(result_dict['max_dy'], dim=0),  # [N]
            'min_dy': torch.cat(result_dict['min_dy'], dim=0),  # [N]
            'sample_indices': torch.tensor(result_dict['sample_indices'])
        }, save_path)
        print(f"Results saved to {save_path}")

    def train(self, train_loader, val_loader):
        best_loss = float('inf')

        for epoch in range(self.config['epochs']):
            train_loss, train_time = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)

            # 记录日志
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(f"best_{self.config['model_name']}.pth")

            # 定期保存
            if epoch % self.config['save_interval'] == 0:
                self.save_model(f"{self.config['model_name']}_epoch{epoch}.pth")

            print(f"Epoch {epoch + 1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Time: {train_time:.1f}s")

    def save_model(self, filename):
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state': self.model.module.state_dict() if isinstance(self.model,
                                                                        nn.DataParallel) else self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }, save_path)

    @classmethod
    def load_model(cls, config, model_config, checkpoint_path):
        trainer = cls(config, model_config)
        checkpoint = torch.load(checkpoint_path)
        if isinstance(trainer.model, nn.DataParallel):
            trainer.model.module.load_state_dict(checkpoint['model_state'])
        else:
            trainer.model.load_state_dict(checkpoint['model_state'])

        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return trainer


# 配置示例
base_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'multi_gpu': True,
    'model_name': 'unet_base',
    'log_dir': './logs',
    'save_dir': './checkpoints',
    'save_interval': 20,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'loss_type': 'huber',
    'epochs': 100
}

# 初始化示例
if __name__ == "__main__":
    # 数据加载
    rootPath = "/home/csh/results20250331"
    train_list = np.load(f"{rootPath}/train_dx_dy.npy")
    test_list = np.load(f"{rootPath}/test_dx_dy.npy")
    feature_stats_path = f"{rootPath}/feature_stats.txt"
    global_stats_path = f"{rootPath}/dx_statistics.txt"

    dataset_train = DataSet_hyper(train_list, feature_stats_path, global_stats_path)
    dataset_test = DataSet_hyper(test_list, feature_stats_path, global_stats_path)

    train_loader = DataLoader(dataset_train,
                              batch_size=base_config['batch_size'],
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    val_loader = DataLoader(dataset_test,
                            batch_size=base_config['batch_size'],
                            num_workers=2)

    # 模型配置
    model_config = ModelConfig(model_size='base')

    # 初始化训练器
    trainer = UNetTrainer(base_config, model_config)

    # 恢复训练示例
    trainer = UNetTrainer.load_model(base_config, model_config, "checkpoints/best_unet_base.pth")

    # 开始训练
    trainer.train(train_loader, val_loader)