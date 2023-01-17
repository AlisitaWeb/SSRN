#
# demo.py
#
import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image
torch.set_printoptions(profile="full")


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, default=r'D:\PT\archive\Testingset10class\dataset\结果\test_96_label',
                        help='image to test')
    parser.add_argument('--out-path', type=str, default=r'D:\PT\archive\Testingset10class\dataset\结果\96', help='mask image to save')
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default=r'D:\PT\超分辨率语义分割\模型保存\10_class\dsrl\128/model_best.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='rockdataset',
                        choices=['pascal', 'coco', 'cityscapes', 'rockdataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=96,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    for name in os.listdir(args.in_path):
        image = Image.open(args.in_path + "/" + name).convert('RGB')

        # image = Image.open(args.in_path).convert('RGB')
        target = Image.open(args.in_path + "/" + name)
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['label'].unsqueeze(0)
        print(tensor_in.shape)



        grid_image = make_grid(decode_seg_map_sequence(tensor_in.detach().cpu().numpy()),
                               3, normalize=False, range=(0, 255))
        save_image(grid_image, args.out_path + "/" + "{}_label.png".format(name[0:-4]))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in in_path.")


if __name__ == "__main__":
    main()

# python demo.py --in-path your_file --out-path your_dst_file
