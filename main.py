from __future__ import print_function
import argparse
from .train import train
from .test import test
from utils import *

parser = argparse.ArgumentParser(description='')

# parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of total epoches')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='number of samples in one batch')# 16
parser.add_argument('--out_dir', dest='output', default='./output', help='directory for checkpoints')
parser.add_argument('--act', dest='activation function', default='leaky', help='one of [relu, leaky, prelu]')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--train_dir', dest='test_dir', default='./data/train', help='directory for testing inputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test', help='directory for testing inputs')

args = parser.parse_args()


def main(_):
    if args.phase != 'train' or args.phase != 'test':
        print('[!] Please Input Phase == train or test')
        exit(0)
    else:
        train(args)


if __name__ == '__main__':
    main()
