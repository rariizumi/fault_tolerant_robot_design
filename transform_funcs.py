# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:30:28 2023

@author: ariizumi
"""

import os
import json
import torch

import numpy as np

# from eagent.model import Model

#%%
class TransformFuncs:
    def __init__(self, survived_act_nodes, survived_obs_nodes,
                 old_max_num_limbs, new_max_num_limbs):
        self.survived_act_nodes = survived_act_nodes
        self.survived_obs_nodes = survived_obs_nodes
        self.old_max_num_limbs = old_max_num_limbs
        self.new_max_num_limbs = new_max_num_limbs
    
    def old_obs_2_new_obs(self, old_obs):
        n = 4 * self.new_max_num_limbs + 11
        new_obs = old_obs[self.survived_obs_nodes]
        new_obs = torch.concat([new_obs, torch.zeros((n - 11 - len(self.survived_obs_nodes)))])
        new_obs = torch.concat([new_obs, old_obs[-11:]])
        return new_obs
    
    def new_obs_2_old_obs(self, new_obs):
        n = 4 * self.old_max_num_limbs + 11
        old_obs = torch.zeros((n, ))
        old_obs[self.survived_obs_nodes] = new_obs[:len(self.survived_obs_nodes)]
        old_obs[-11:] = new_obs[-11:]
        return old_obs
    
    def old_act_2_new_act(self, old_act):
        n = 2 * self.new_max_num_limbs
        new_act = torch.zeros((n, ))
        new_act[:len(self.survived_act_nodes)] = old_act[self.survived_act_nodes]
        return new_act
    
    def new_act_2_old_act(self, new_act):
        n = 2 * self.old_max_num_limbs
        old_act = torch.zeros((n, ))
        old_act[self.survived_act_nodes] = new_act[:len(self.survived_act_nodes)]
        return old_act

if __name__ == '__main__':
    #%% parameters
    old_params_name = os.path.join('log', 'transform_test', 'parameter_best.json')
    new_params_name = os.path.join('log', 'transform_test', 'parameter_transformed.json')
    
    with open(old_params_name, 'r') as f:
        old_params = json.load(f)
    
    with open(new_params_name, 'r') as f:
        new_params = json.load(f)
        
    #%% common configs
    old_cfg_name = os.path.join('log', 'transform_test', 'cfg_old.json')
    new_cfg_name = os.path.join('log', 'transform_test', 'cfg_new.json')
    
    with open(old_cfg_name, 'r') as f:
        old_cfg = json.load(f)
    
    with open(new_cfg_name, 'r') as f:
        new_cfg = json.load(f)
    
    old_max_num_limbs = old_cfg['max_num_limbs']
    new_max_num_limbs = new_cfg['max_num_limbs']
    
    #%% survived nodes
    survived_act_nodes = np.loadtxt(os.path.join('log', 'transform_test', 'survived_act_nodes.csv'), delimiter=',')
    survived_obs_nodes = np.loadtxt(os.path.join('log', 'transform_test', 'survived_obs_nodes.csv'), delimiter=',')
    survived_nodes = (survived_act_nodes / 2)[::2]
    
    #%% instantiate
    t = TransformFuncs(survived_act_nodes, survived_obs_nodes, old_max_num_limbs, new_max_num_limbs)
    
    #%% obs
    new_obs = torch.zeros((4*new_max_num_limbs+11, ))
    new_obs[:len(survived_obs_nodes)] = torch.tensor(np.arange(0, len(survived_obs_nodes)).astype(np.float32))
    new_obs[-11:] = torch.ones((11, ))
    
    old_obs = torch.zeros((4*old_max_num_limbs+11, ))
    old_obs[survived_obs_nodes] = torch.tensor(np.arange(0, len(survived_obs_nodes)).astype(np.float32))
    old_obs[-11:] = torch.ones((11, ))
    
    diff_obs_old2new = new_obs - t.old_obs_2_new_obs(old_obs)
    diff_obs_new2old = old_obs - t.new_obs_2_old_obs(new_obs)
    
    #%% act
    new_act = torch.zeros((2*new_max_num_limbs, ))
    new_act[:len(survived_act_nodes)] = torch.tensor(np.arange(0, len(survived_act_nodes)).astype(np.float32))
    
    old_act = torch.zeros((2*old_max_num_limbs, ))
    old_act[survived_act_nodes] = torch.tensor(np.arange(0, len(survived_act_nodes)).astype(np.float32))
    
    # NOTE: we asign 0 to unrelated elements
    diff_act_old2new = new_act - t.old_act_2_new_act(old_act)
    diff_act_new2old = old_act - t.new_act_2_old_act(new_act)
    
    #%% all should be zero
    print('|diff_obs_old2new| = ', np.linalg.norm(diff_obs_old2new))
    print('|diff_obs_new2old| = ', np.linalg.norm(diff_obs_new2old))
    print('|diff_act_old2new| = ', np.linalg.norm(diff_act_old2new))
    print('|diff_act_new2old| = ', np.linalg.norm(diff_act_new2old))
    