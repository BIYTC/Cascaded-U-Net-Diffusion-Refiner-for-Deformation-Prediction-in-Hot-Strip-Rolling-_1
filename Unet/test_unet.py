from unet_main import *
from dataset import *

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

    batch_size = 2

    dataset_test = DataSet_hyper(test_list, feature_stats_path, global_stats_path)
    test_loader = DataLoader(dataset_test,
                            batch_size=batch_size,
                            num_workers=2)

    # 模型配置
    model_config = ModelConfig(model_size='base')

    # 初始化训练器
    trainer = UNetTrainer(base_config, model_config)

    # 恢复训练示例
    trainer = UNetTrainer.load_model(base_config, model_config, "checkpoints/best_unet_base.pth")

    # 开始训练
    trainer.test(test_loader)

