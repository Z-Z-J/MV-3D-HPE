# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:45:20 2023

@author: ZZJ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.spatial_transformer import SpatialTransformer
from model.temporal_transformer import TemporalTransformer

#-------------------------------------------------------------------------------
class AttShrink(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        
        self.shrink = nn.Conv2d(in_channels, out_channels, 1)
    
    def set_bn_momentum(self, momentum):
        self.bn.momentum = momentum
    
    def forward(self, x):
        B, C, T, V = x.shape
        V_view = int(V**(0.5))
        x = self.drop(self.relu(self.bn(self.conv(x))))
      
        x = self.shrink(x)
        x = x.view(B, -1, T, V_view, V_view)
        return x


class AgShrink(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
                
        self.shrink = nn.Conv2d(in_channels, out_channels, 1)
    
    def set_bn_momentum(self, momentum):
        self.bn.momentum = momentum
    
    def forward(self, x):
        B, C, T, V = x.shape
        V_view = int(V**(0.5))
        x = self.drop(self.relu(self.bn(self.conv(x))))
        
        x = self.shrink(x)
        x = x.view(B, -1, T, V_view, V_view)
        return x


class VAL(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.25, momentum=0.1, num_joints=17):
        super().__init__()
        
        
        self.expand_conv = nn.Conv2d(in_channels, in_channels, (1, 1), bias=False)
        self.expand_bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        
        # Pos embedding
        self.pos_emb_conv = nn.Conv2d(int(num_joints*2), in_channels, 1, bias=True)        
        
        # Res block
        self.num_layers = 1
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
           conv_layers.append(nn.Conv2d(in_channels, in_channels, 1, bias = False, ))
           bn_layers.append(nn.BatchNorm2d(in_channels, momentum=momentum))
           conv_layers.append(nn.Conv2d(in_channels, in_channels, 1, bias = False))
           bn_layers.append(nn.BatchNorm2d(in_channels, momentum=momentum))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
        # Ag / Att
        self.ag_shrink = AgShrink(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
        self.att_shrink = AttShrink(in_channels=in_channels, out_channels=1, dropout=dropout)
    
    def set_bn_momentum(self, momentum):
        
        self.expand_bn.momentum = momentum
        
        self.ag_shrink.set_bn_momentum(momentum)
        
        self.att_shrink.set_bn_momentum(momentum)
        
        for bn in self.bn_layers:
            bn.momentum = momentum
    
    def forward(self, x, pos_2d, rotation=None):
        """
        Args:
            x: (B,C,T,N)
            pos_2d: (B,T,J,C,N_view)
        """
        B, _, T, N = x.shape
        N_view = int(N**(0.5))
        
        pos_2d = pos_2d[:,:,:,:2,:].permute(0, 2, 3, 1, 4).contiguous()  # (B, J, 2, T, N_view)
        pos_2d = pos_2d.view(B, -1, T, N_view)
        rel_pos = pos_2d[:, :, :, :, None] - pos_2d[:, :, :, None, :]  # (B, J*C, T, N_view, N_view)
        rel_pos = rel_pos.view(B, -1, T, N)
        
        # Pos embedding
        pos_emb = self.pos_emb_conv(rel_pos)
        
        # linear
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        # 
        x = x + pos_emb
        
        # Res block
        K = 2
        for i in range(self.num_layers):
            res = x
            x = self.drop(self.relu(self.bn_layers[K * i](self.conv_layers[K * i](x))))
            x = self.drop(self.relu(self.bn_layers[K * i + 1](self.conv_layers[K * i +1](x))))
            x = res + x
        
        # ag
        ag = self.ag_shrink(x)
        
        # att
        att = self.att_shrink(x)
    
        return ag, att


class FuseView(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.25, num_joints=17):
        super().__init__()
        
        
        self.pose_model = VAL(in_channels = in_channels, out_channels = out_channels, dropout = dropout, num_joints = num_joints)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
                
        self.W = nn.Parameter(torch.zeros(size=(2, out_channels, out_channels), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
    
    def set_bn_momentum(self, momentum):
        self.pose_model.set_bn_momentum(momentum)
    
    def forward(self, x, pos_2d, rotation=None):
        
        x = rearrange(x, 'b t j c n -> b (j c) t n')
        B, C, T, N = x.shape
        B, T, J, _, N = pos_2d.shape
        
        f = x.clone()  # B,C,T,N
      
        x1 = x.view(B, C, T, N, 1).repeat(1, 1, 1, 1, N)
        x2 = x.view(B, C, T, 1, N).repeat(1, 1, 1, N, 1)  
        x = torch.cat((x1, x2), dim=1)
        x = x.view(B, C*2, T, N*N)
       
        ag, att = self.pose_model(x, pos_2d, rotation)  # ag: [B,C,T,N,N]  att: [B,1,T,N,N]
        
        f_self = torch.einsum('bctn, cd -> bdtn', f, self.W[0]) 
        f_self = f_self.view(B,-1,T,1,N).repeat(1,1,1,N,1)
        
        f_tran = torch.einsum('bctn, cd -> bdtn', f, self.W[1])
        f_tran = f_tran.view(B,-1,T,1,N).repeat(1,1,1,N,1)
        
        f_tran = f_tran * ag
        
        E = torch.eye(N, dtype=torch.float).view(1,1,1,N,N).to(x.device)

        f_conv = E * f_self + (1-E) * f_tran  # B, C, T, N, N
        
        att = F.softmax(att, dim=-1) 

        f_fuse = f_conv * att

        f_fuse = torch.sum(f_fuse, dim=-1)
        
        f_fuse = f_fuse + f
        f_fuse = f_fuse.view(B, J, -1, T, N).permute(0, 3, 1, 2, 4)
        return f_fuse
 
    
class VideoMultiViewModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        
        
        # Spatial
        self.spatial_3d_model = SpatialTransformer(num_joints=cfg.joints, in_chans=cfg.in_chans-1, embed_dim_ratio=cfg.embed_chans, 
                                          depth=cfg.spatial_layers, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
        
        self.spatial_2d_model = SpatialTransformer(num_joints=cfg.joints, in_chans=cfg.in_chans, embed_dim_ratio=cfg.embed_chans, 
                                          depth=cfg.spatial_layers, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
        
        
        # View
        fuse_chans = cfg.joints * cfg.embed_chans
        
        self.fuse_3d_model = FuseView(in_channels=fuse_chans*2, out_channels=fuse_chans, dropout = cfg.dropout, num_joints=cfg.joints)
         
        self.fuse_2d_model = FuseView(in_channels=fuse_chans*2, out_channels=fuse_chans, dropout = cfg.dropout, num_joints=cfg.joints)
     
        # Temporal
        self.temporal_3d_model = TemporalTransformer(num_frames=cfg.frames, embed_dim_ratio=cfg.embed_chans, 
                                                  depth=cfg.temporal_layers, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
        
        self.temporal_2d_model = TemporalTransformer(num_frames=cfg.frames, embed_dim_ratio=cfg.embed_chans, 
                                                  depth=cfg.temporal_layers, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
        
        # Head
        self.shrink_3d = nn.Linear(cfg.embed_chans, cfg.out_chans)
        
         
        self.shrink_2d = nn.Linear(cfg.embed_chans, cfg.out_chans-1)

    def set_bn_momentum(self, momentum):
        self.fuse_3d_model.set_bn_momentum(momentum)
        self.fuse_2d_model.set_bn_momentum(momentum)
        
        
    def forward(self, pos_2d, rotation = None):
        B, T, J, C, N = pos_2d.shape
         
        pos_2d = pos_2d.contiguous()
    
        # 2D
        f_2d = self.spatial_2d_model(pos_2d) 
        f_2d = self.fuse_2d_model(f_2d, pos_2d, rotation) 
        f_2d = self.temporal_2d_model(f_2d)       
        # 读出
        out_f_2d = rearrange(f_2d, 'b t j c n -> (b t j n) c')
        out_2d = self.shrink_2d(out_f_2d)
        out_2d = rearrange(out_2d, '(b t j n) d -> b t j d n', t=T, j=J, n=N)
        
        # 3D
        f_3d = self.spatial_3d_model(out_2d)
        f_3d = self.fuse_3d_model(f_3d, pos_2d, rotation)
        f_3d = self.temporal_3d_model(f_3d)
        # 读出
        out_f_3d = rearrange(f_3d, 'b t j c n -> (b t j n) c')
        out_3d = self.shrink_3d(out_f_3d)
        out_3d = rearrange(out_3d, '(b t j n) d -> b t j d n', t=T, j=J, n=N)
        
        return out_3d, out_2d
    
    
if __name__ == "__main__":
    from common.opt import opts    
    opt = opts().parse()

    Net = VideoMultiViewModel(opt)
    
    pos_2d = torch.randn(2,27,17,3,4)
    
    a, b = Net(pos_2d)

    model_params = 0
    for parameter in Net.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
        













