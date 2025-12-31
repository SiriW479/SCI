import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os
import rawpy

batch_w = 600
batch_h = 400


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task, black_level=512, white_level=16383):
        self.low_img_dir = img_dir
        self.task = task
        self.black_level = black_level  # 黑电平，根据相机型号调整
        self.white_level = white_level  # 白电平，根据相机型号调整
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                # 支持常见的 raw 格式
                if name.lower().endswith(('.dng', '.arw', '.nef', '.cr2', '.raw')):
                    self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

    def load_raw_image(self, file):
        """加载 raw 图像并转换为 RGGB 4 通道格式"""
        with rawpy.imread(file) as raw:
            # 获取 raw 数据
            raw_data = raw.raw_image.copy().astype(np.float32)
            
            # 减去黑电平
            raw_data = np.maximum(raw_data - self.black_level, 0)
            
            # 归一化到 [0, 1]
            raw_data = raw_data / (self.white_level - self.black_level)
            raw_data = np.clip(raw_data, 0, 1)
            
            # 获取 Bayer 模式
            # 大多数相机使用 RGGB 模式
            h, w = raw_data.shape
            
            # 将 Bayer 数据重组为 4 通道 (R, G1, G2, B)
            rggb = np.zeros((h // 2, w // 2, 4), dtype=np.float32)
            rggb[:, :, 0] = raw_data[0::2, 0::2]  # R
            rggb[:, :, 1] = raw_data[0::2, 1::2]  # G1
            rggb[:, :, 2] = raw_data[1::2, 0::2]  # G2
            rggb[:, :, 3] = raw_data[1::2, 1::2]  # B
            
            return rggb

    def __getitem__(self, index):

        low = self.load_raw_image(self.train_low_data_names[index])

        h = low.shape[0]
        w = low.shape[1]
        
        # 随机裁剪（训练时）
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))
        
        if self.task != 'test':
            low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        low = np.asarray(low, dtype=np.float32)
        # 转换为 (C, H, W) 格式
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count
