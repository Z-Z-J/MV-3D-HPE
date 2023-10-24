# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:51:34 2023

@author: ZZJ
"""
import os
import logging
from tqdm import tqdm

import numpy as np

import torch
import torch.utils.data
import torch.optim as optim

from common.opt import opts
from common.load_data_h36m import Fusion

from common.loss_pos import define_mpjpe_error_list, mpjpe_cal, test_mpjpe_calculation

from common.utils import define_actions, AccumLoss, get_varialbe, print_error

from model.svt_former_2 import VideoMultiViewModel

from common.set_seed import *


def val(opt, actions, test_loader, model):
    model_test = model['pos_test']
    with torch.no_grad():
        return step('test', opt, actions, test_loader, model_test)
    
def step(split, opt, actions, dataloader, model_pos, optimizer=None, epoch=None):
    
    loss_all = {
        'loss_p3d': AccumLoss(),
        }
       
    action_3d_mpjpe_sum = define_mpjpe_error_list(actions)  

    if split == 'train':
        model_pos.train()
    else:
        model_pos.eval()
    
    t = tqdm(dataloader, 0)
    for i, data in enumerate(t):
        cameras, input_poses, action = data
        inputs_poses, cameras = get_varialbe(split, [input_poses, cameras])
    
        # gt_2d
        inputs_2d_gt = inputs_poses[...,:2,:]
        # pre_2d
        inputs_2d_pre = inputs_poses[...,2:4,:]
        # gt_3d
        inputs_3d_gt = inputs_poses[...,4:7,:]
      
    
        # inference
        if split == 'train':
            out_3D = model_pos(inputs_2d_gt)
        else:
            inputs_2d_gt, inputs_3d_gt, out_3D = input_augmentation(inputs_2d_gt, inputs_3d_gt, model_pos)
        
        # gt traj
        inputs_traj_gt = inputs_3d_gt[:,:,0:1].clone()
        inputs_3d_gt[:,:,:1] = 0
        
        # loss 3d
        loss_p3d = mpjpe_cal(inputs_3d_gt, out_3D)
    
        loss_p3d_np = loss_p3d.detach().cpu().numpy()
       
        B, T, J, _, N = inputs_2d_gt.shape
        Np = B * T * J * N
      
        loss_all['loss_p3d'].update(loss_p3d_np * Np, Np)
    
        # loss prompting
        t.set_description('3d_mpjpe({0:,.4f})'.format(loss_all["loss_p3d"].avg)) 
        t.refresh()
        
        
        action_3d_mpjpe_sum = test_mpjpe_calculation(out_3D, inputs_3d_gt, action, action_3d_mpjpe_sum)
      
    mpjpe_3d = print_error(action_3d_mpjpe_sum, 'MPJPE', opt.train)
      
    return mpjpe_3d
        

def input_augmentation(input_2D_gt, input_3D_gt, model):
    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]
        
    
    input_2D_gt_non_flip = input_2D_gt[:, 0] 
    input_2D_gt_flip = input_2D_gt[:,1]
    
    input_3D_gt_non_flip = input_3D_gt[:, 0]

    output_3D_flip = model(input_2D_gt_flip)
    
    output_3D_flip[:, :, :, 0] *= -1 
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]  
    
    
    # Non flip
    output_3D_non_flip = model(input_2D_gt_non_flip)
    
    output_3D = (output_3D_non_flip + output_3D_flip) / 2.

    
    return input_2D_gt_non_flip, input_3D_gt_non_flip, output_3D
        

if __name__ == '__main__':
    # ----------------------------opt & gpu & seed & log-----------------------
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    print(opt)
    
    set_seed()
    """
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    """
    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    
    # ----------------------------------dataset--------------------------------
    actions = define_actions(opt.actions)
    if opt.train:
        train_data = Fusion(opt=opt, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    
    test_data = Fusion(opt=opt, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    
    # ----------------------------------Model----------------------------------
    model = {}
    model['pos_test'] = VideoMultiViewModel(opt).cuda()
    
    # ----------------------------------Load-----------------------------------
    model_dict = model['pos_test'].state_dict()
    if opt.resume or opt.test:
        chk_filename = opt.previous_dir 
        pre_dict = torch.load(chk_filename)
        
        model['pos_test'].load_state_dict(pre_dict['model'], strict=False)
    

    # ---------------------------------Test-----------------------------------
    mpjpe_p3d = val(opt, actions, test_dataloader, model)
    print('mpjpe_p3d: %.2f' % (mpjpe_p3d))
          
