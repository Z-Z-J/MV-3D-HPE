# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:54:45 2023

@author: ZZJ
"""
import torch
import numpy as np

from einops import rearrange


def make_intrinsic_mat(cam_intrinsic_params):
    """
    cam_intrinsic_params: [b,c,n]
    """
    B, C, N = cam_intrinsic_params.shape
    device = cam_intrinsic_params.device
    cam_intrinsic_params = rearrange(cam_intrinsic_params, 'b c n -> (b n) c')
    
    K = torch.eye(3,dtype=torch.float32).to(device)
    K = K.unsqueeze(0).repeat(B*N,1,1)
    f = torch.sqrt(cam_intrinsic_params[:,0] * cam_intrinsic_params[:,1])
    K[:,0,0] = f
    K[:,1,1] = f
    K[:,0,2] = cam_intrinsic_params[:,2]
    K[:,1,2] = cam_intrinsic_params[:,3]
    
    K = K.view(B,N,3,3)
    return K


def make_rotation_mat(pos_3d_gt, index):
    """
    pos_3d_gt: (b,t,j,c,n)
    """
    B, T, J, C, N = pos_3d_gt.shape
    
    source = pos_3d_gt[...,index:index+1].clone()
    source = source.repeat(1,1,1,1,N)
    target = pos_3d_gt.clone()
    
    source = rearrange(source, 'b t j c n -> (b n) (j t) c')
    target = rearrange(target, 'b t j c n -> (b n) (j t) c')
    
    muX = torch.mean(target, dim=1, keepdims=True)
    muY = torch.mean(source, dim=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY
    
    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    X0 = X0 / normX
    Y0 = Y0 / normY
    
    H = torch.matmul(X0.permute(0, 2, 1), Y0)
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.permute(0, 2, 1)
    R = torch.matmul(V, U.permute(0, 2, 1))
    R = R.permute(0,2,1)  # [b,3,3]
    R = R.view(B,N,3,3)
    
    return R


def make_translation_mat(traj_gt, rotation, index):
    """
    traj_gt : (b,t,1,c,n)
    rotation: (b,n,3,3)
    """
    B, T, J, C, N = traj_gt.shape
    
    source = traj_gt[...,index:index+1].clone()
    source = source.repeat(1,1,1,1,N)
    target = traj_gt.clone()
    
    source = rearrange(source, 'b t j c n -> (b n) (j t) c')
    target = rearrange(target, 'b t j c n -> (b n) (j t) c')
    rotation = rearrange(rotation, 'b n c d -> (b n) c d')
    
    source = source.unsqueeze(-1)
    out = torch.einsum('bcd, bndh -> bnch', rotation, source)
    out = out.squeeze(-1)
    translation = torch.mean(target - out, dim=-2)
    
    translation = translation.view(B,N,3)
    return translation


def make_extrinsic_mat(pos_3d_gt, traj_gt, index):
    """
    pos_3d_gt: (b,t,j,c,n)
    traj_gt : (b,t,1,c,n)
    """
    B,T,J,C,N = pos_3d_gt.shape
    device = pos_3d_gt.device
    
    R = make_rotation_mat(pos_3d_gt, index)  # b n 3, 3
    t = make_translation_mat(traj_gt, R, index)
    
    exi_mat = torch.zeros([B,N,3,4], dtype=torch.float32).to(device)
    exi_mat[:,:,:,:3] = R
    exi_mat[:,:,:,-1] = t
    
    return exi_mat

"""
def triangulation(pts, pmat):
    '''
    pts: (batch, njoints, nview, 2)
    pmat: (nview, 3, 4) or (batch, nview, 3, 4)
    '''
    dev = pts.device

    batch, njoint, nview = pts.shape[0:3]#(batch, njoints, nview, 2)

    if len(pmat.shape) == 3:
        pmat = pmat.to(dev).view(1, nview, 3, 4).repeat(batch * njoint, 1, 1, 1) #(batch * njoints, nview, 3, 4)
    elif len(pmat.shape) == 4:
        pmat = pmat.to(dev).view(batch, 1, nview, 3, 4).repeat(1, njoint, 1, 1, 1).view(batch*njoint, nview, 3, 4) #(batch * njoints, nview, 3, 4)
    pts_compact = pts.view(batch*njoint, nview, 2, 1)
    A = pmat[:,:,2:3] * pts_compact #(batch*njoint, nview, 2, 4)
    A -= pmat[:,:,:2]
   
    A = A.view(-1, 2 * nview, 4)
    A_np = A.cpu().numpy()
    try:
        u, d, vt = np.linalg.svd(A_np)  # vt (batch*njoint, 4, 4)
        Xs = vt[:,-1,0:3]/vt[:,-1,3:]
    except np.linalg.LinAlgError:
        Xs = np.zeros((batch*njoint, 3), dtype=np.float32)
    except FloatingPointError:
        # print(vt[:,-1,3:])
        div = vt[:,-1,3:]
        div[div==0] = float('inf')
        Xs = vt[:,-1,0:3]/vt[:,-1,3:]

    # convert ndarr to tensor
    Xs = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    Xs = Xs.view(batch, njoint, 3)
    return Xs
"""
def triangulation(pts, pmat):
    '''
    pts: (batch, njoints, nview, 2)
    pmat: (nview, 3, 4) or (batch, nview, 3, 4)
    '''
    dev = pts.device

    batch, njoint, nview = pts.shape[0:3]#(batch, njoints, nview, 2)

    if len(pmat.shape) == 3:
        pmat = pmat.to(dev).view(1, nview, 3, 4).repeat(batch * njoint, 1, 1, 1) #(batch * njoints, nview, 3, 4)
    elif len(pmat.shape) == 4:
        pmat = pmat.to(dev).view(batch, 1, nview, 3, 4).repeat(1, njoint, 1, 1, 1).view(batch*njoint, nview, 3, 4) #(batch * njoints, nview, 3, 4)
    pts_compact = pts.view(batch*njoint, nview, 2, 1)
    A = pmat[:,:,2:3] * pts_compact #(batch*njoint, nview, 2, 4)
    A -= pmat[:,:,:2]
   
    A = A.view(-1, 2 * nview, 4)
    #A_np = A.cpu().numpy()
    try:
        u, d, vt = torch.linalg.svd(A)  # vt (batch*njoint, 4, 4)
        Xs = vt[:,-1,0:3]/vt[:,-1,3:]
    except np.linalg.LinAlgError:
        Xs = torch.zeros((batch*njoint, 3), dtype=np.float32)
    except FloatingPointError:
        # print(vt[:,-1,3:])
        div = vt[:,-1,3:]
        div[div==0] = float('inf')
        Xs = vt[:,-1,0:3]/vt[:,-1,3:]

    # convert ndarr to tensor
    #Xs = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    Xs = Xs.view(batch, njoint, 3)
    return Xs


def p2d_cam3d(pos_2d, pos_3d_gt, traj_gt, cameras):
    """
    pos_2d: (b,t,j,c,n)
    pos_3d_gt: (b,t,j,c,n)
    traj_gt: (b,t,1,c,n)
    cameras: (b,c,n)
    """
    B, T, J, C, N = pos_2d.shape
    pos_2d = rearrange(pos_2d, 'b t j c n -> (b t) j n c')
    pos_2d = pos_2d.contiguous()
    
    intri_mat = make_intrinsic_mat(cameras)  # B,N,3,3 
    traj_w3d_list = []
    for i in range(N):
        extri_mat = make_extrinsic_mat(pos_3d_gt, traj_gt, i)  # B,N,3,4
        prj_mat = torch.einsum('bnkj,bnjc -> bnkc', intri_mat, extri_mat)  # B,N,3,4
        prj_mat = prj_mat.view(B,1,N,3,4).repeat(1,T,1,1,1)
        prj_mat = prj_mat.view(-1,N,3,4).contiguous()
        trj_w3d = triangulation(pos_2d, prj_mat)
        traj_w3d_list.append(trj_w3d.view(B,T,J,3))
    
    out = torch.stack(traj_w3d_list, dim=-1)
    return out







