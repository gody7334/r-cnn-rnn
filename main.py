from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import _init_paths
from control.train import Train
from control.evaluate import Evaluate
from control.prepare_dataset import PrepareDataset
from utils.config import global_config

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--mode', help='mode should be one of "train" "eval" "inference"', required=True)
args = vars(parser.parse_args())

def main():
    if args['mode'] == "train":
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        Train().run()
    elif args['mode'] == "eval":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        Evaluate().run()

# def prepare_dataset():
#     global_config.assign_config()
#     preparedb=PrepareDataset()
#     imdb, roidb = preparedb.combined_roidb('voc_2007_trainval')
#     print('{:d} roidb entries'.format(len(roidb)))

if __name__ == '__main__':
    main()
    # prepare_dataset()
