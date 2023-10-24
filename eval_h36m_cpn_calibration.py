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
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from common.opt import opts
from common.load_data_h36m import Fusion

from common.loss_pos import define_mpjpe_error_list, mpjpe_cal, test_mpjpe_calculation


from common.utils import define_actions, AccumLoss, get_varialbe, print_error

from model.svt_former_1 import VideoMultiViewModel

from common.ray import encoder_ray, decoder_ray


from common.trangulation import p2d_cam3d

from common.set_seed import *

import random
def train(opt, actions, train_loader, model, optimizer, epoch):
    model_train = model['pos_train'] 
    return step('train', opt, actions, train_loader, model_train, optimizer, epoch)

def val(opt, actions, test_loader, model):
    model_test = model['pos_test']
    with torch.no_grad():
        return step('test', opt, actions, test_loader, model_test)
    
def step(split, opt, actions, dataloader, model_pos, optimizer=None, epoch=None):
    
    loss_all = {
        'loss_p3d': AccumLoss(),  
        'loss_traj': AccumLoss(),
        }
        
      
    action_3d_mpjpe_sum = define_mpjpe_error_list(actions) 
  
    action_traj_mpjpe_sum = define_mpjpe_error_list(actions)
  

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
        # vis
        vis = inputs_poses[...,7:8,:]
        # gt_3d
        inputs_3d_gt = inputs_poses[...,4:7,:]
        #
        #cameras = cameras[...,index]

       
        inputs_ray_pre, inputs_ray_gt, cameras, inputs_3d_gt, out_2D = input_augmentation(inputs_2d_pre, inputs_2d_gt, cameras, vis, inputs_3d_gt, model_pos)
        

        # gt traj
        inputs_traj_gt = inputs_3d_gt[:,:,0:1].clone()
        inputs_3d_gt[:,:,:1] = 0
        inputs_2d_gt = decoder_ray(inputs_ray_gt, cameras)
        
        
        out_cam_3D = p2d_cam3d(out_2D, inputs_3d_gt, inputs_traj_gt, cameras)
        out_traj = out_cam_3D[:,:,0:1,:,:]
        out_3D = out_cam_3D - out_traj
        
      
        
        # loss 2d/3d
        loss_p3d = mpjpe_cal(inputs_3d_gt, out_3D)
        loss_traj = mpjpe_cal(inputs_traj_gt, out_traj)
    
        
      
        loss_p3d_np = loss_p3d.detach().cpu().numpy()
        loss_traj_np = loss_traj.detach().cpu().numpy()
     
        
        B, T, J, _, N = inputs_3d_gt.shape
        Np = B * T * J * N
        Ntraj = B * T * 1 * N
     
      
        loss_all['loss_p3d'].update(loss_p3d_np * Np, Np)
        loss_all['loss_traj'].update(loss_traj_np * Ntraj, Ntraj) 
     
        
        # loss prompting
        t.set_description('3d_mpjpe({0:,.4f}), traj_mpjpe({1:,.4f})'.format(loss_all["loss_p3d"].avg, loss_all["loss_traj"].avg))
        t.refresh()
        
       
            
        action_3d_mpjpe_sum = test_mpjpe_calculation(out_3D, inputs_3d_gt, action, action_3d_mpjpe_sum)
        action_traj_mpjpe_sum = test_mpjpe_calculation(out_traj, inputs_traj_gt, action, action_traj_mpjpe_sum)
            
    
    mpjpe_3d = print_error(action_3d_mpjpe_sum, 'MPJPE', opt.train)
    mpjpe_traj = print_error(action_traj_mpjpe_sum, 'MPJPE', opt.train) 
        
    return mpjpe_3d, mpjpe_traj
        

def input_augmentation(input_2D_pre, input_2D_gt, cameras, vis, input_3D_gt, model):
    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]
        
    input_2D_pre_non_flip = input_2D_pre[:, 0] 
    input_2D_pre_flip = input_2D_pre[:, 1]      # B,T,N,C,V
    input_2D_gt_non_flip = input_2D_gt[:, 0] 
    
    cameras_non_flip = cameras[:,0]
    cameras_flip = cameras[:,1]
    
    input_ray_pre_non_flip = encoder_ray(input_2D_pre_non_flip, cameras_non_flip)
    input_ray_pre_flip = encoder_ray(input_2D_pre_flip, cameras_flip)
    input_ray_gt_non_flip = encoder_ray(input_2D_gt_non_flip, cameras_non_flip)
    
    input_3D_gt_non_flip = input_3D_gt[:, 0]
    
    vis_non_flip = vis[:,0]
    vis_flip = vis[:,1]
    
    inp_flip = torch.cat((input_2D_pre_flip, vis_flip), dim=-2)
    
    output_2D_flip = model(inp_flip)
    
    output_2D_flip[:, :, :, 0] *= -1 
    output_2D_flip[:, :, joints_left + joints_right, :] = output_2D_flip[:, :, joints_right + joints_left, :]  
    
    
    # Non flip
    inp_non_flip = torch.cat((input_2D_pre_non_flip, vis_non_flip), dim=-2)
    output_2D_non_flip = model(inp_non_flip)
    
    output_2D = (output_2D_non_flip + output_2D_flip) / 2.

    
    return input_ray_pre_non_flip, input_ray_gt_non_flip, cameras_non_flip, input_3D_gt_non_flip, output_2D
        

if __name__ == '__main__':
    # ----------------------------opt & gpu & seed & log-----------------------
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    print(opt)
    
    #set_seed()
    """
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    """
 
    # ----------------------------------dataset--------------------------------
    actions = define_actions(opt.actions)
    
    test_data = Fusion(opt=opt, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    # ----------------------------------Model----------------------------------
    model = {}
    model['pos_test'] =  nn.DataParallel(VideoMultiViewModel(opt)).cuda()
    
    # ----------------------------------Load-----------------------------------
    model_dict = model['pos_test'].state_dict()
    if opt.resume or opt.test:
        chk_filename = opt.previous_dir 
        pre_dict = torch.load(chk_filename)
        
        model['pos_test'].load_state_dict(pre_dict['model'], strict=False)
    
    # ---------------------------------Test-----------------------------------
 
    mpjpe_p3d, mpjpe_traj = val(opt, actions, test_dataloader, model)
        
    print('mpjpe_p3d: %.2f, mpjpe_traj: %.2f' % (mpjpe_p3d, mpjpe_traj))
           
