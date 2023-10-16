# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:00:29 2022

@author: ZZJ
"""
import torch

def encoder_ray(points2d, cam):
    """
    points2d: [b,t,j,2,n]
    cam: [b,4,n]
    """
    B, T, J, C, N = points2d.shape
    
    f = torch.sqrt(cam[:,0:1,:] * cam[:,1:2,:])
    f = f.unsqueeze(1).unsqueeze(1)
    
    cx = cam[:,2:3,:].unsqueeze(1).unsqueeze(1)
    cy = cam[:,3:4,:].unsqueeze(1).unsqueeze(1)
   
    ray_x = (points2d[:,:,:,0:1,:] - cx) / f
    ray_y = (points2d[:,:,:,1:2,:] - cy) / f
    
    ray = torch.cat([ray_x, ray_y], dim=-2)
    
    return ray



def decoder_ray(ray, cam):
    """
    ray: [b,t,j,2,n]
    cam: [b,4,n]
    """
    B, T, J, C, N = ray.shape
    
    f = torch.sqrt(cam[:,0:1,:] * cam[:,1:2,:])
    f = f.unsqueeze(1).unsqueeze(1)
    
    cx = cam[:,2:3,:].unsqueeze(1).unsqueeze(1)
    cy = cam[:,3:4,:].unsqueeze(1).unsqueeze(1)
   
    points2d_x = ray[:,:,:,0:1,:] * f + cx
    points2d_y = ray[:,:,:,1:2,:] * f + cy
    
    points2d = torch.cat([points2d_x, points2d_y], dim =-2)
    return points2d

