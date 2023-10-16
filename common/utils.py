# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:29:37 2023

@author: ZZJ
"""

import os
import torch
from torch.autograd import Variable

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

def define_actions( action ):

    actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise( ValueError, "Unrecognized action: %s" % action )

    return [action]

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        
        
def print_error(action_error_sum, error_type, is_train):
    mean_error = print_error_action(action_error_sum, error_type, is_train)

    return mean_error


def print_error_action(action_error_sum, error_type, is_train):
    mean_error_each = 0.0
    mean_error_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", error_type))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each = action_error_sum[action].avg * 1000.0
        mean_error_all.update(mean_error_each, 1)

        if is_train == 0:
            print("{0:>7.2f}".format(mean_error_each))

    if is_train == 0:
        print("{0:<12} {1:>7.2f}".format("Average", mean_error_all.avg))
    
    return mean_error_all.avg


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        