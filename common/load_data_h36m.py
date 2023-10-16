# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:02:00 2023

@author: ZZJ
"""
import numpy as np

import torch.utils.data as data

import pickle

from common.generator import ChunkedGenerator

from common.h36m_dataset import Human36mCamera

class Fusion(data.Dataset):
    def __init__(self, opt, is_train=True):
        
        self.root_path = opt.root_path
        self.p2d_type = opt.p2d_type
        
        self.train = is_train
        
        self.train_subject_list = opt.subjects_train.split(',')
        self.test_subject_list = opt.subjects_test.split(',')
        
        self.used_cameras = opt.cameras_train if is_train else opt.cameras_test
        
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        
        self.pad = opt.pad
        
        self.step = opt.step
        
        self.test_aug = opt.test_augmentation
        
        self.cameras = Human36mCamera()
        
        if self.train:
            self.keypoints, self.vis_score = self.prepare_data(self.root_path, self.p2d_type, self.train_subject_list)
            self.actions_train, self.cameras_train, self.poses_train = self.fetch(self.train_subject_list, is_test=False)
            
            self.generator = ChunkedGenerator(opt.batch_size, self.actions_train, self.cameras_train, self.poses_train,
                                              chunk_length=1, pad=self.pad, causal_shift=0,
                                              shuffle=True, augment=True,
                                              kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left, joints_right=self.joints_right,
                                              step=opt.step)
            
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints, self.vis_score = self.prepare_data(self.root_path, self.p2d_type, self.test_subject_list)
            self.actions_test, self.cameras_test, self.poses_test = self.fetch(self.test_subject_list, is_test=True)
            
            self.generator = ChunkedGenerator(opt.batch_size, self.actions_test, self.cameras_test, self.poses_test,
                                              chunk_length=1, pad=self.pad, causal_shift=0,
                                              shuffle=False, augment=False,
                                              kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left, joints_right=self.joints_right,
                                              step=1)
            
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))
            
    def prepare_data(self, root_path, p2d_type, subject_list):
        # ------------------P2D & P2D_GT & P3D---------------------------------
        keypoints = {}
        for sub in subject_list:
        
            if p2d_type == 'cpn' or p2d_type == 'gt':
                data_pth = root_path + 'h36m_sub{}.npz'.format(sub[1:])
            elif p2d_type == 'ada_fuse':
                data_pth = root_path + 'h36m_sub{}_ada_fuse.npz'.format(sub[1:])
            
            keypoint = np.load(data_pth, allow_pickle=True)
            keypoints_metadata = keypoint['metadata'].item()
            keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
            keypoints[sub] = keypoint['positions_2d'].item()[sub]
        
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = self.kps_left, self.kps_right
        
        # --------------------------Vis----------------------------------------
        if p2d_type == 'cpn' or p2d_type == 'gt':
            vis_path = root_path + 'score.pkl'
        elif p2d_type == 'ada_fuse':
            vis_path = root_path + 'vis_ada.pkl'
        
        vis_score = pickle.load(open(vis_path, 'rb'))
        
        return keypoints, vis_score
    
    def fetch(self, subjects, is_test=False):
    
        out_poses_view1 = []
        out_poses_view2 = []
        out_poses_view3 = []
        out_poses_view4 = []
        
        out_actions = []
        
        out_cameras = []
        
        for subject in subjects:
            
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startwith(a) and len(action.split(a)[1]) < 3:
                            found = True
                            break
                    if not found:
                        continue
                
                out_actions.append(action)
                
                pos = self.keypoints[subject][action]  # [view0,view1,view2,view3]
                
                n_frames = pos[0].shape[0]
                vis_name_1 = '{}_{}.{}'.format(subject, action, 0)
                vis_name_2 = '{}_{}.{}'.format(subject, action, 1)
                vis_name_3 = '{}_{}.{}'.format(subject, action, 2)
                vis_name_4 = '{}_{}.{}'.format(subject, action, 3)
                vis_score_cam0 = self.vis_score[vis_name_1][:n_frames][...,np.newaxis]
                vis_score_cam1 = self.vis_score[vis_name_2][:n_frames][...,np.newaxis]
                vis_score_cam2 = self.vis_score[vis_name_3][:n_frames][...,np.newaxis]
                vis_score_cam3 = self.vis_score[vis_name_4][:n_frames][...,np.newaxis]
                if vis_score_cam3.shape[0] != vis_score_cam2.shape[0]:
                    vis_score_cam2 = vis_score_cam2[:-1]
                    vis_score_cam1 = vis_score_cam1[:-1]
                    vis_score_cam0 = vis_score_cam0[:-1]
                    for i in range(4):
                        pos[i] = pos[i][:-1]
                
                if is_test == True and action == 'Walking' and pos[0].shape[0] == 1612:
                    out_poses_view1.append(np.concatenate((pos[0][1:], vis_score_cam0[1:]), axis =-1))
                    out_poses_view2.append(np.concatenate((pos[1][1:], vis_score_cam1[1:]), axis =-1))
                    out_poses_view3.append(np.concatenate((pos[2][1:], vis_score_cam2[1:]), axis =-1))
                    out_poses_view4.append(np.concatenate((pos[3][1:], vis_score_cam3[1:]), axis =-1))
                else:
                    out_poses_view1.append(np.concatenate((pos[0], vis_score_cam0), axis =-1))
                    out_poses_view2.append(np.concatenate((pos[1], vis_score_cam1), axis =-1))
                    out_poses_view3.append(np.concatenate((pos[2], vis_score_cam2), axis =-1))
                    out_poses_view4.append(np.concatenate((pos[3], vis_score_cam3), axis =-1))
   
        final_pose = []
        if 0 in self.used_cameras:
            final_pose.append(out_poses_view1)
        if 1 in self.used_cameras:
            final_pose.append(out_poses_view2)
        if 2 in self.used_cameras:
            final_pose.append(out_poses_view3)
        if 3 in self.used_cameras:
            final_pose.append(out_poses_view4)
        
        out_cameras = self.cameras.get_intrinsic_params(self.used_cameras)
        
        return out_actions, out_cameras, final_pose
    
    def __len__(self):
        return len(self.generator.pairs)
    
    def __getitem__(self, index):
        seq_name, start_index, end_index, flip = self.generator.pairs[index]
        
        cameras, poses, actions = self.generator.get_batch(seq_name, start_index, end_index, flip)
        
        if self.train == False and self.test_aug:
            cameras_aug, poses_aug, _ = self.generator.get_batch(seq_name, start_index, end_index, flip=True)
            poses = np.concatenate((np.expand_dims(poses, axis=0), np.expand_dims(poses_aug, axis=0)), 0)
            cameras = np.concatenate((np.expand_dims(cameras, axis=0), np.expand_dims(cameras_aug, axis=0)), 0)
        
        return cameras, poses, actions
    
                
                
                
                
                
                
                
                
                
                
                
                
        
        
        
        
        
        
            