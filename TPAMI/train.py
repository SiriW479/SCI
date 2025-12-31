import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from model import *
from multi_read_data import MemoryFriendlyLoader


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=3000, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='Rebuttal/000/', help='location of the data corpus')
parser.add_argument('--data_path', type=str, default='../../raw', help='path to raw images folder')
parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio of training data (default: 0.7)')
parser.add_argument('--black_level', type=int, default=512, help='black level for raw images')
parser.add_argument('--white_level', type=int, default=16383, help='white level for raw images')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
csv_path = args.save+'/csv_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'train.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    """保存 raw 图像为 4 通道 RGGB 格式的 numpy 文件"""
    raw_numpy = tensor[0].cpu().float().numpy()  # (4, H, W)
    # 保存为 .npy 格式
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
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    model = Network(stage=args.stage)

    model = model.cuda()
    optimizer_a = torch.optim.Adam(model.ha.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    optimizer_b = torch.optim.Adam(list(model.hb.parameters()) + list(model.calibrate.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)

    # 加载所有 raw 图像路径
    import glob
    all_raw_files = []
    for ext in ['*.dng', '*.DNG', '*.arw', '*.ARW', '*.nef', '*.NEF', '*.cr2', '*.CR2', '*.raw', '*.RAW']:
        all_raw_files.extend(glob.glob(os.path.join(args.data_path, ext)))
    all_raw_files.sort()
    
    # 按 7:3 划分训练集和测试集
    num_total = len(all_raw_files)
    num_train = int(num_total * args.train_ratio)
    
    train_files = all_raw_files[:num_train]
    test_files = all_raw_files[num_train:]
    
    logging.info(f"Total images: {num_total}, Training: {num_train}, Testing: {num_total - num_train}")
    
    # 创建临时目录存放划分后的文件路径
    train_dir = os.path.join(args.save, 'train_list')
    test_dir = os.path.join(args.save, 'test_list')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存文件列表
    with open(os.path.join(train_dir, 'train_files.txt'), 'w') as f:
        for file in train_files:
            f.write(file + '\n')
    with open(os.path.join(test_dir, 'test_files.txt'), 'w') as f:
        for file in test_files:
            f.write(file + '\n')

    TrainDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='train', 
                                        black_level=args.black_level, white_level=args.white_level)
    # 使用训练集索引
    train_indices = list(range(num_train))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test',
                                       black_level=args.black_level, white_level=args.white_level)
    # 使用测试集索引
    test_indices = list(range(num_train, num_total))
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=0, generator=torch.Generator(device = 'cuda'))

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1, sampler=test_sampler,
        num_workers=0, generator=torch.Generator(device = 'cuda'))

    total_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            

            total_step += 1
            input = Variable(input, requires_grad=True).cuda()

            _, loss1 , loss2 , loss3  = model._loss_Jiaoti(input)
            
            if total_step %10 < 7:
                for param in model.ha.parameters():
                    param.requires_grad = True
                for param in model.hb.parameters():
                    param.requires_grad = False
                for param in model.calibrate.parameters():
                    param.requires_grad = False
                optimizer_a.zero_grad()

                loss = loss1
                loss.backward()
                optimizer_a.step()
            else:
                for param in model.ha.parameters():
                    param.requires_grad = False
                for param in model.hb.parameters():
                    param.requires_grad = True
                for param in model.calibrate.parameters():
                    param.requires_grad = True
                optimizer_b.zero_grad()

                loss = (loss2 + loss3) 
                loss.backward()
                optimizer_b.step()
            
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            losses.append(loss.item())
            logging.info('train-epoch-{:0>3d}-step-{:0>5d}  Tloss:{:<8.4f} loss1:{:<8.4f} loss2:{:<8.4f} loss3:{:<8.4f}'\
                         .format(epoch, batch_idx, loss,loss1,loss2,loss3))

            if total_step % 500 == 0 and total_step != 0:
                logging.info('train %03d %f', epoch, loss)
                model.eval()
                with torch.no_grad():
                    for _, (input, image_name) in enumerate(test_queue):
                        input = Variable(input, volatile=True).cuda()
                        image_name = image_name[0].split('/')[-1].split('.')[0]
                        _, ref_list, _, _= model(input)
                        for ii in range(1):
                            u_name = '{}_{}_{}.png'.format(image_name, total_step, ii)
                            u_path = image_path + '/' + u_name
                            save_images(ref_list[ii], u_path)
                model.train()
                logging.info('train-epoch %03d %f', epoch, np.average(losses))
                utils.save(model, os.path.join(model_path, 'weights_%d_%d.pt' % (epoch, total_step)))



if __name__ == '__main__':
    main()
