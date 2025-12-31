import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel

from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='./data/raw',
                    help='location of the raw data')
parser.add_argument('--save_path', type=str, default='./results', help='location to save results')
parser.add_argument('--model', type=str, default='./weights/weights_1_3500.pt', help='model weights path')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--black_level', type=int, default=512, help='black level for raw images')
parser.add_argument('--white_level', type=int, default=16383, help='white level for raw images')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test',
                                   black_level=args.black_level, white_level=args.white_level)

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    """保存 raw 图像为 4 通道 RGGB 格式的 numpy 文件"""
    raw_numpy = tensor[0].cpu().float().numpy()  # (4, H, W)
    # 保存为 .npy 格式（raw 数据）
    np.save(path.replace('.png', '.npy'), raw_numpy)
    
    # 可选：将 RGGB 转换为 RGB 用于可视化
    # 简单的去马赛克：取两个 G 通道的平均
    r = raw_numpy[0]
    g = (raw_numpy[1] + raw_numpy[2]) / 2.0
    b = raw_numpy[3]
    rgb = np.stack([r, g, b], axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    im = Image.fromarray(np.clip(rgb * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(args.model)
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('/')[-1].split('.')[0]
            i, r = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)



if __name__ == '__main__':
    main()
