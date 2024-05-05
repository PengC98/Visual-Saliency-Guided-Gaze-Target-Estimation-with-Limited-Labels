import os
import argparse
import yaml
from main_util import *
from train import *

from PIL import Image

parser = argparse.ArgumentParser(description='Train the gaze target estimation network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu_ids', default='1', dest='gpu_ids')

parser.add_argument('--mode', default='test', choices=['train', 'test', 'train_leaky'], dest='mode')
parser.add_argument('--train_continue', default='on', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--dir_checkpoint', default='/DATA/cp/gaze/checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='/DATA/cp/gaze/log', dest='dir_log')
parser.add_argument('--scope', default='gte', dest='scope')
parser.add_argument('--name_data', type=str, default='GazeFollow', dest='name_data')
parser.add_argument('--dir_data', default='/DATA/cp/gazefollow_extended', dest='dir_data')

parser.add_argument('--num_epoch', type=int,  default=10, dest='num_epoch')

parser.add_argument('--batch_size', type=int, default=20, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=0.0001, dest='lr_G')


parser.add_argument('--accumulation_steps', type=int, default=500, dest='accumulation_steps')

parser.add_argument('--beta1', default=0.9, dest='beta1')

parser.add_argument('--num_freq_disp', type=int,  default=500, dest='num_freq_disp')

parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')

PARSER = Parser(parser)

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

if __name__ == '__main__':

    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()
    path = 'parameter.yaml'
    Dict = read_yaml(path)

    TRAINER = Train(ARGS)
    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()








