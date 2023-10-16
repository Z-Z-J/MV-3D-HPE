# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:00:42 2023

@author: ZZJ
"""
import torch
from common.utils import AccumLoss


def define_mpjpe_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: AccumLoss() for i in range(len(actions))})
    return error_sum

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape  # (B,T,J,C,V)
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 2))

def test_mpjpe_calculation(predicted, target, action, error_sum):
    error_sum = error_by_action(predicted, target, action, error_sum)
    return error_sum

def error_by_action(predicted, target, action, action_error_sum):
    """
    predicted : shape B T J C V
    """
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    view_num = predicted.size(-1)
    
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 2), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name].update(torch.mean(dist).item()*batch_num*frame_num*view_num, batch_num*frame_num*view_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name].update(torch.mean(dist[i]).item()*frame_num*view_num, frame_num*view_num)
            
    return action_error_sum
