from diffuseModel import DiffusionModel
import numpy as np
from diffusionUtils import *
from dataset import *
from Unet.unet_main import UNetTrainer, Unet_ModelConfig
# 2万条数据带验证功能和验证结果记录

# 参数规模调节示例：
# 大型配置（约2.5亿参数）
class LargeConfig(ModelConfig):
    # 保持原有配置但确保通道数合理
    channel_mult = [4, 8, 16, 32, 64, 128]  # 实际通道数：8 * 4=32, 32 * 8=256等
    num_res_blocks = 4
    time_dim = 1024
    attn_layers = [3, 4, 5]
    use_adaGN = True


# 小型配置（约2000万参数）
class SmallConfig(ModelConfig):
    channel_mult = [1, 2, 4, 8]
    num_res_blocks = 2
    time_dim = 256
    attn_layers = []

# 使用示例：
# model = DiffusionModel(LargeConfig())

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Initialize model
        self.model = DiffusionModel(SmallConfig).to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1 and config['multi_gpu']:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        # Initialize diffusion utils
        self.diffusion = DiffusionUtils(
            timesteps=config['timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            device=self.device
        )
        rootPath = "/home/csh/results20250331"
        train_list = np.load(f"{rootPath}/train_dx_dy.npy")
        val_list = np.load(f"{rootPath}/test_dx_dy.npy")
        feature_stats_path = f"{rootPath}/feature_stats.txt"
        global_stats_path = f"{rootPath}/dx_statistics.txt"
        self.train_dataset = DataSet_hyper(train_list, feature_stats_path, global_stats_path)
        self.val_dataset = DataSet_hyper(val_list, feature_stats_path, global_stats_path)
        # Initialize dataset and dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.75,
            patience=5,
            min_lr=1e-8,
            verbose=True
        )

        # Loss function
        self.loss_fn = config['loss_fn']

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_val_loss = float('inf')

        # Load checkpoint if exists
        if config['resume'] and os.path.exists(config['model_dir'] + config['checkpoint_path']):
            self.load_checkpoint(config['model_dir'] + config['checkpoint_path'])

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (features, target, _, _, _, _) in enumerate(progress_bar):
            features = features.to(self.device)
            target = target.to(self.device)

            # Sample random timesteps
            t = torch.randint(0, self.diffusion.timesteps, (features.size(0),), device=self.device).long()

            # Calculate loss
            loss = self.diffusion.p_losses(
                self.model,
                target,
                features,
                t,
                loss_type=self.loss_fn
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            self.writer.add_scalar('train/batch_loss', loss.item(),
                                   self.current_epoch * len(self.train_loader) + batch_idx)

        # Calculate epoch metrics
        epoch_loss /= len(self.train_loader)
        epoch_time = time.time() - start_time

        # Update scheduler
        self.scheduler.step(epoch_loss)

        # Log to tensorboard
        self.writer.add_scalar('train/epoch_loss', epoch_loss, self.current_epoch)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.current_epoch)

        return epoch_loss, epoch_time

    def train(self):
        print("Starting training...")

        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            # Train one epoch
            epoch_loss, epoch_time = self.train_epoch()
            current_lr = self.scheduler.get_last_lr()[0]
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config['epochs']}, Loss: {epoch_loss:.4f}, LR:{current_lr}, Time: {epoch_time:.2f}s")

            # Save checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint()

            # Save checkpoint periodically
            if (epoch + 1) % self.config['save_interval'] == 0:
                # print("开始验证...")  # 验证太慢了，先不验证了
                # val_loss = self.validate()
                # print(f"第{self.current_epoch}轮验证loss为{val_loss}")
                # if val_loss < self.best_val_loss:
                #     print(f"第{self.current_epoch}轮验证loss为{val_loss}小于当前最佳验证loss{self.best_val_loss}")
                #     self.best_val_loss = val_loss
                #     self.save_checkpoint(f"{self.config['model_dir']}checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(f"{self.config['model_dir']}checkpoint_new_epoch.pt")

        print("Training completed!")
        self.writer.close()

    def validate(self, ddim=False, eta=0.0):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating")

            for features, target, *_ in progress_bar:
                features = features.to(self.device)
                target = target.to(self.device)

                # Generate samples
                samples = self.diffusion.sample(
                    self.model,
                    features,
                    (features.size(0), 2, 21, 51),
                    self.device,
                    ddim=ddim,
                    eta=eta
                )

                # Calculate loss
                loss = F.mse_loss(samples, target)
                total_loss += loss.item()

                progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(self.val_loader)
        print(f"Average test loss: {avg_loss:.4f}")
        # Log to tensorboard
        self.writer.add_scalar('validate/avg_loss', avg_loss, self.current_epoch)
        return avg_loss

    def save_checkpoint(self, filename=None):
        if filename is None:
            filename = f"{self.config['model_dir']}{self.config['checkpoint_path']}"

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model,
                                                                             nn.DataParallel) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']

        print(f"Checkpoint loaded from {path}. Resuming from epoch {self.current_epoch + 1}")


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        # Initialize model
        self.model = DiffusionModel(SmallConfig).to(self.device)
        # 初始化Unet模型
        model_config = Unet_ModelConfig(model_size='base')
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
        # 加载模型
        self.unet_model = UNetTrainer.load_model(base_config, model_config, "./checkpoints/best_unet_base.pth")

        # Load model weights
        self.load_model(config['model_dir'])

        # Initialize diffusion utils
        self.diffusion = DiffusionUtils(
            timesteps=config['timesteps'],
            ddim_timesteps = config['ddim_timesteps'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            device=self.device
        )

        # Initialize dataset for ground truth comparison
        rootPath = "/home/csh/results20250331"
        test_list = np.load(f"{rootPath}/test_dx_dy.npy")
        feature_stats_path = f"{rootPath}/feature_stats.txt"
        global_stats_path = f"{rootPath}/dx_statistics.txt"
        self.test_dataset = DataSet_hyper(test_list, feature_stats_path, global_stats_path)
        # Initialize dataset and dataloader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

    def load_model(self, path):
        checkpoint = torch.load(path + self.config['checkpoint_path'], map_location=self.device)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model loaded from {path}")

    def test_sample(self, features, ddim=False, eta=0.0):
        self.model.eval()

        with torch.no_grad():
            features = features.to(self.device)
            shape = (features.size(0), 2, 21, 51)

            # Generate samples
            samples = self.diffusion.sample(
                self.model,
                shape,
                self.device,
                ddim=ddim,
                eta=eta
            )

            return samples.cpu()

    def evaluate(self, ddim=False, eta=0.0):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")

            for features, target, _, _, _, _ in progress_bar:
                features = features.to(self.device)
                target = target.to(self.device)

                # Generate samples
                samples = self.diffusion.sample(
                    self.model,
                    features,
                    (features.size(0), 2, 21, 51),
                    self.device,
                    ddim=ddim,
                    eta=eta
                )

                # Calculate loss
                loss = F.mse_loss(samples, target)
                total_loss += loss.item()

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.test_loader)
        print(f"Average test loss: {avg_loss:.4f}")
        return avg_loss

    def visulize_output(self, ddim=False, eta=0.0):
        self.model.eval()
        self.unet_model.model.eval()

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
            progress_bar = tqdm(enumerate(self.test_loader), desc="Testing")

            for i, (features, target, max_dx, min_dx, max_dy, min_dy) in progress_bar:
                if i % 200 == 0:
                    print(f"当前进行第{i}个样本预测：")
                    features = features.to(self.device)
                    target = target.to(self.device)
                    b = target.shape[0]
                    # Generate samples
                    outputs = self.diffusion.sample(
                        self.model, self.unet_model.model,
                        features,
                        (features.size(0), 2, 21, 51),
                        self.device,
                        ddim=ddim,
                        eta=eta, denoise_step=self.config['denoise_step']
                    )
                    loss = F.mse_loss(outputs, target)
                    print(f"当前样本loss：{loss}")
                    # 新增部分：收集结果
                    batch_indices = [i * self.test_loader.batch_size + j for j in range(b)]  # 计算样本全局索引
                    result_dict['preds'].append(outputs.cpu().detach())  # 存储为CPU张量
                    result_dict['inputs'].append(features.cpu().detach())
                    result_dict['targets'].append(target.cpu().detach())
                    result_dict['max_dx'].append(max_dx.cpu().detach())
                    result_dict['min_dx'].append(min_dx.cpu().detach())
                    result_dict['max_dy'].append(max_dy.cpu().detach())
                    result_dict['min_dy'].append(min_dy.cpu().detach())
                    result_dict['sample_indices'].extend(batch_indices)

                    break
                # 新增部分：保存结果到文件
        save_path = "test_results_Cascade_ddpm_2.pth"
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

    def timed_inference_test(self, method='ddpm', ddim_timesteps=1000, eta=0.0):
        """执行带时间统计的推理测试"""
        # 创建新的扩散工具实例
        # diffusion = DiffusionUtils(
        #     timesteps=timesteps,
        #     beta_start=self.config['beta_start'],
        #     beta_end=self.config['beta_end'],
        #     device=self.device
        # )
        diffusion = DiffusionUtils(
            timesteps=self.config['timesteps'],
            ddim_timesteps = ddim_timesteps,
            beta_start=self.config['beta_start'],
            beta_end=self.config['beta_end'],
            device=self.device
        )

        # 随机选择10个样本（不重复）
        total_samples = len(self.test_dataset)
        indices = torch.randperm(total_samples)[:10]
        subset = torch.utils.data.Subset(self.test_dataset, indices)
        test_loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            # num_workers=self.config['num_workers'],
            num_workers=0,
            pin_memory=True
        )

        results = []

        for idx, (features, target, *_) in enumerate(test_loader):
            # 数据转移到设备
            features = features.to(self.device)
            target = target.to(self.device)

            # 预热GPU（排除第一次运行的初始化时间）
            if idx == 0:
                _ = diffusion.sample(self.model, features, (1, 2, 21, 51),
                                     self.device, ddim=(method == 'ddim'), eta=eta)

            # 精确计时
            if self.device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            else:
                start_time = time.time()

            # 执行推理
            pred = diffusion.sample(
                self.model,
                features,
                (1, 2, 21, 51),
                self.device,
                ddim=(method.lower() == 'ddim'),
                eta=eta
            )

            # 结束计时
            if self.device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000  # 毫秒转秒
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time

            # 计算指标
            mse = F.mse_loss(pred, target).item()
            mae = F.l1_loss(pred, target).item()

            results.append((elapsed_time, mae, mse))
            print(f"Sample {idx + 1}: Time={elapsed_time:.4f}s, MAE={mae:.4f}, MSE={mse:.4f}")

        # 生成文件名
        fname = f"./Results/{method}_steps_{ddim_timesteps}"
        if method.lower() == 'ddim':
            fname += f"_eta_{eta:.2f}"
        fname += "_results.txt"

        # 保存结果
        with open(fname, 'w') as f:
            f.write("Time(s)\tMAE\tMSE\n")
            for t, mae, mse in results:
                f.write(f"{t:.6f}\t{mae:.6f}\t{mse:.6f}\n")

        print(f"\n测试结果已保存到 {fname}")
        return fname

    def visualize_denoise_process(self, sample_idx=0, ddim=False, eta=0.0,
                                  output_dir="denoise_process", steps_to_save=50, hybrids_steps=30):
        """
        可视化单个样本的去噪过程（所有步骤保存在单个文件中）
        :param sample_idx: 要可视化的样本索引
        :param steps_to_save: 要保存的中间步骤数（均匀间隔）
        :param output_dir: 输出目录
        :param hybrids_steps: 保留hybrids_steps步用ddim
        """
        self.model.eval()
        self.unet_model.model.eval()

        os.makedirs(output_dir, exist_ok=True)

        # 获取指定样本
        features, target, max_dx, min_dx, max_dy, min_dy = self.test_dataset[sample_idx]
        features = features.unsqueeze(0).to(self.device)

        # 初始化结果字典（扩展结构以包含所有步骤）
        result_dict = {
            'all_preds': [],  # 新增：存储所有步骤的预测 [steps, 2, 21, 51]
            'inputs': features.cpu(),
            'targets': target.unsqueeze(0),
            'max_dx': max_dx,
            'min_dx': min_dx,
            'max_dy': max_dy,
            'min_dy': min_dy,
            'timesteps': []  # 新增：记录每个步骤对应的时间步
        }

        # 自定义采样过程
        b = 1
        # img = torch.randn((b, 2, 21, 51), device=self.device)
        img = self.unet_model.model(features)
        # 将最原始的噪声也保存一下
        result_dict['all_preds'].append(img.cpu())
        result_dict['timesteps'].append(500)

        if ddim:
            step_interval = self.diffusion.ddim_timesteps // steps_to_save
            with torch.no_grad():
                # 均匀步长
                # step_ratio = self.diffusion.timesteps // self.diffusion.ddim_timesteps
                # _t_list = list(range(0, self.diffusion.timesteps, step_ratio))  #缺一个499代的
                # _t_list.append(499)  # 填补最初一步的噪声
                # 用指数步长，去噪前期步长短后面步长长
                # ratios = np.linspace(0, 1, self.diffusion.ddim_timesteps)
                # indices = (self.diffusion.timesteps-1) * (ratios ** 0.3)
                # _t_list = np.floor(indices).astype(int).tolist()
                # 用二次间隔步长，去噪前期步长间隔长，后期步长短，比cos步数下降的更快
                ratios = (np.arange(self.diffusion.ddim_timesteps) / self.diffusion.ddim_timesteps) ** 2
                _t_list = (self.diffusion.timesteps * ratios).astype(int).tolist()
                # for i in tqdm(reversed(range(0, self.diffusion.ddim_timesteps+1)), desc='sampling loop time step',  # 均匀步长的
                for i in tqdm(reversed(range(0, self.diffusion.ddim_timesteps)), desc='sampling loop time step',  # 指数步长的
                              total=self.diffusion.ddim_timesteps):
                    if i >= hybrids_steps:  # 用ddim进行快速降噪
                        _t = _t_list[i]
                        t = torch.full((b,), _t, device=self.device, dtype=torch.long)  # ddim采用跳步采样
                        img = self.diffusion.p_sample_ddim(self.model, img, features, t, i, eta=eta)
                    else:  # 用ddpm进行精修
                        t = torch.full((b,), i, device=self.device, dtype=torch.long)
                        img = self.diffusion.p_sample(self.model, img, features, t, i)
                    # 记录指定步骤
                    if i % step_interval == 0 or i == 0 or i == self.diffusion.ddim_timesteps - 1 or i <= 10:
                        result_dict['all_preds'].append(img.cpu())
                        result_dict['timesteps'].append(i)
        else:
            step_interval = self.diffusion.timesteps // steps_to_save
            with torch.no_grad():
                for i in tqdm(reversed(range(0, self.diffusion.timesteps)), desc='Denoising Process'):
                    # 尝试一下从200代开始去噪
                    if i < self.config['denoise_step']:
                        t = torch.full((b,), i, device=self.device, dtype=torch.long)
                        # DDPM 采样
                        img = self.diffusion.p_sample(self.model, img, features, t, i)
                        # 记录指定步骤
                        if i % step_interval == 0 or i == 0 or i == self.diffusion.timesteps-1 or i<=50:
                            result_dict['all_preds'].append(img.cpu())
                            result_dict['timesteps'].append(i)

        # 转换列表为张量 [steps, 2, 21, 51]
        result_dict['all_preds'] = torch.concat(result_dict['all_preds'], dim=0)

        # 保存文件（与visulize_output相同的格式）
        filename = f"sample_{sample_idx}_process.pth"
        torch.save(result_dict, os.path.join(output_dir, filename))
        print(f"Denoising process saved to {os.path.join(output_dir, filename)}")