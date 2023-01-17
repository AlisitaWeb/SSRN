import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import csv
from modeling.deeplab import *
from utils.metrics import Evaluator
from dataset import VOCSegmentation
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


parser = argparse.ArgumentParser(description="PyTorch DSRL Training")
parser.add_argument('--backbone', type=str, default='mobilenet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--dataset', type=str, default='rockdataset_10',
                    choices=['pascal', 'coco', 'cityscapes', 'invoice'],
                    help='dataset name (default: pascal)')
parser.add_argument('--crop-size', type=int, default=128,
                    help='crop image size')
parser.add_argument('--num_classes', type=int, default=11,
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--batch-size', type=int, default=4)
args = parser.parse_args()

kwargs = {'num_workers': 1, 'pin_memory': True}
setup_seed(1)
dataset = VOCSegmentation(split='val')
val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
evaluator = Evaluator(11)
model = DeepLab(num_classes=11,
                backbone='mobilenet',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)

ckpt = torch.load(r'D:\PT\超分辨率语义分割\模型保存\1\dsrl\96/model_best.pth.tar', map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model = model.cuda()
model.eval()
evaluator.reset()
tbar = tqdm(val_dataloader, desc='\r')

'''
f = open(r'D:\PT\超分辨率语义分割\模型保存\1\dsrl\96/dsrl.csv', mode='a', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(f, fieldnames=['Acc', 'Acc_class', 'mIoU', 'FWIoU'])

f = open(r'D:\PT\超分辨率语义分割\模型保存\1\dsrl\96/dsrl_acc_class.csv', mode='a', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(f, fieldnames=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

'''
f = open(r'D:\PT\超分辨率语义分割\模型保存\1\dsrl\96/dsrl_iou_val.csv', mode='a', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(f, fieldnames=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

csv_writer.writeheader()

for i, sample in enumerate(tbar):
    image, target = sample['image'], sample['label']
    input_img = torch.nn.functional.interpolate(image, size=[i // 2 for i in image.size()[2:]], mode='bilinear',
                                                align_corners=True)
    input_img, image, target = input_img.cuda(), image.cuda(), target.cuda()
    with torch.no_grad():
        output, _, _, _ = model(input_img)
    pred = output.data.cpu().numpy()
    target = target.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    evaluator.add_batch(target, pred)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    '''
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    dic = {
        'Acc': Acc,
        'Acc_class': Acc_class,
        'mIoU': mIoU,
        'FWIoU': FWIoU,
    }

    dic = {
        '0': Acc_class[0],
        '1': Acc_class[1],
        '2': Acc_class[2],
        '3': Acc_class[3],
        '4': Acc_class[5],
        '5': Acc_class[6],
        '6': Acc_class[7],
        '7': Acc_class[8],
        '8': Acc_class[9],
        '9': Acc_class[10],
    }
    '''
    dic = {
        '0': mIoU[0],
        '1': mIoU[1],
        '2': mIoU[2],
        '3': mIoU[3],
        '4': mIoU[5],
        '5': mIoU[6],
        '6': mIoU[7],
        '7': mIoU[8],
        '8': mIoU[9],
        '9': mIoU[10],
    }

    csv_writer.writerow(dic)
