# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:53:52 2023

@author: ZZJ
"""
import numpy as np


class ChunkedGenerator():
    def __init__(self, batch_size, actions, cameras, poses,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234, augment=True,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, step=1):
        
        # DB
        tmp = []
        num_cam = len(poses)
        self.VIEWS = range(num_cam)
    
        for i in range(len(poses[0])):  # num of videos
            n_frames = 1000000000
            for n in range(num_cam):  # num of cams
                if poses[n][i].shape[0] < n_frames:
                    n_frames = poses[n][i].shape[0]
            
            for n in range(num_cam):
                poses[n][i] = poses[n][i][:n_frames]
            
            temp_pos = poses[0][i][..., np.newaxis]
            for j in range(1, num_cam):
                temp_pos = np.concatenate((temp_pos, poses[j][i][..., np.newaxis]), axis=-1)
           
            tmp.append(temp_pos)
        self.db = tmp
        self.cameras = cameras
        
        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        self.saved_index = {}
        start_index = 0
        
        for i in range(len(poses[0])):  # num of videos
            n_chunks = (poses[0][i].shape[0] + chunk_length -1) // chunk_length
            offset = (n_chunks * chunk_length - poses[0][i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            keys = np.tile(np.array((i, actions[i])).reshape([1,2]), (len(bounds -1), 1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector))
            if augment:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector))
            
            end_index = start_index + poses[0][i].shape[0]
            self.saved_index[i] = [start_index,end_index]
            start_index = start_index + poses[0][i].shape[0]
        
        pairs = pairs[::step]
        
        # Params
        self.num_batches = (len(pairs) + batch_size -1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.batch = np.empty((batch_size, chunk_length + 2*pad, poses[0][0].shape[-2], poses[0][0].shape[-1], num_cam))
        self.batch_cam = np.empty((batch_size, cameras.shape[-2], num_cam))
    
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
    
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            # print('*********************************')
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def get_batch(self, seq_i, start_index, end_index, flip):
        video_index, action = seq_i  
        start = start_index - self.pad - self.causal_shift
        end = end_index + self.pad - self.causal_shift
   
        seq = self.db[int(video_index)].copy()
        
        low = max(start, 0)
        high = min(end, seq.shape[0])
        pad_left = low - start
        pad_right = end - high
        
        if pad_left != 0 or pad_right != 0:
            self.batch = np.pad(seq[low:high], ((pad_left, pad_right), (0, 0), (0, 0), (0,0)), 'edge')
        else:
            self.batch = seq[low:high]
        
        self.batch_cam = self.cameras.copy()
        
        if flip:
            # p2d_gt p2d_pre p3d vis
            self.batch[:,:,0] *= -1
            self.batch[:,:,2] *= -1
            self.batch[:,:,4] *= -1
            
            self.batch[:,self.kps_left+self.kps_right] = self.batch[:,self.kps_right+self.kps_left]
  
            self.batch_cam[2] *=-1  
  
        return self.batch_cam, self.batch, action
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
