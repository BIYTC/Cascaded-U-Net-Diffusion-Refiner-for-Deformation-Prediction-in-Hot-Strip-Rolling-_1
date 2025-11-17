import torch
from torch import nn, einsum
import torch.nn.functional as F
import glob2
import numpy as np


class DataSet_hyper(torch.utils.data.Dataset):
    def __init__(self, file_list, feature_stats_path, global_stats_path):
        self.file_list=file_list
        # 2. 加载统计信息
        # 特征标准化参数
        with open(feature_stats_path, 'r') as f:
            _ = f.readline()  # 跳过标题行
            self.feature_means = list(map(float, f.readline().strip().split('\t')))
            self.feature_stds = list(map(float, f.readline().strip().split('\t')))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_dir = self.file_list[idx]
        folder_name = sample_dir.split('/')[-1]  # 假设路径使用反斜杠，20250319，在linux里是/分隔

        # 加载输入特征
        input_path = f"{sample_dir}/{folder_name}_input.txt"
        with open(input_path, 'r') as f:
            _ = f.readline()  # 跳过标题行
            features = list(map(float, f.readline().strip().split('\t')))

        # 标准化输入特征
        features = (torch.FloatTensor(features) - torch.FloatTensor(self.feature_means)) / torch.FloatTensor(
            self.feature_stds)

        # 加载目标值（位移）
        dx_path = f"{sample_dir}/{folder_name}_dx_matrix.txt"
        dx_matrix = np.loadtxt(dx_path)
        max_dx = np.max(dx_matrix)
        min_dx = np.min(dx_matrix)
        # 加载目标值（角度）
        dy_path = f"{sample_dir}/{folder_name}_dy_matrix.txt"
        dy_matrix = np.loadtxt(dy_path)
        max_dy = np.max(dy_matrix)  # 这里就是dy
        min_dy = np.min(dy_matrix)

        # 最大最小归一化
        target_dx = (dx_matrix - min_dx)/(max_dx - min_dx)  # 归一化后的dx
        target_dy = (dy_matrix - min_dy) / (max_dy - min_dy)  # 归一化后的dy
        # target_dy = (dy_matrix - min_dy) / (max_dy - min_dy)  # 考虑到角度分布不均，先标准化再归一化

        # 归位-1到1区间
        target_dx = (target_dx * 2) - 1
        target_dy = (target_dy * 2) - 1

        # 需要转为float 32
        target_dx = target_dx.astype(np.float32)
        target_dy = target_dy.astype(np.float32)

        # 合并位移和角度
        target = torch.from_numpy(np.stack([target_dx, target_dy], axis=0))
        return features, target, max_dx, min_dx, max_dy, min_dy
