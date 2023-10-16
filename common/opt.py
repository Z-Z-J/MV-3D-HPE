# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:29:25 2023

@author: ZZJ
"""

import argparse
import os
import math
import time

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def init(self):
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('--p2d_type', type=str, default='cpn')
        self.parser.add_argument('--root_path', type=str, default='./dataset/')
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--resume', action='store_true')
        self.parser.add_argument('--actions', type=str, default='*')
        
        self.parser.add_argument('--gpu', type=str, default='0', help='assign the gpu(s) to use')
        self.parser.add_argument('--batch_size', type=int, default=200, help='can be changed depending on your machine')
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('--frames', type=int, default=27)
        self.parser.add_argument('--joints', type=int, default=17)
        self.parser.add_argument('--in_chans', type=int, default=3)
        self.parser.add_argument('--embed_chans', type=int, default=32)
        self.parser.add_argument('--out_chans', type=int, default=3)
        self.parser.add_argument('--dropout', type=float, default=0.1)
        self.parser.add_argument('--pad', type=int, default=13)
        self.parser.add_argument('--step', type=int, default=1)
        
        self.parser.add_argument('--spatial_layers', type=int, default=2)
        self.parser.add_argument('--temporal_layers', type=int, default=2)
        
        self.parser.add_argument('--nepoch', type=int, default=60)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('-lrd', '--lr_decay', type=float, default=0.9)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=80)
        self.parser.add_argument('--previous_best_threshold', type=float, default=math.inf)
    
        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--checkpoint', type=str, default='')
    
    
    
    def parse(self):
        self.init()
        
        self.opt = self.parser.parse_args()
        
        if self.opt.test:
            self.opt.train = 0
        
        self.opt.pad = (self.opt.frames-1) // 2
        
        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'
        
        self.opt.cameras_train = [0,1,2,3]
        self.opt.cameras_test = [0,1,2,3]
        
        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')
            self.opt.checkpoint = 'checkpoint/' + logtime + '%d'%(self.opt.frames)
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                   if not name.startswith('_'))
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
        
        return self.opt
    
    
        
        
        
