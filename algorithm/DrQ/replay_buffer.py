import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-2], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.others = np.empty((capacity, 4), dtype=np.float32)
        self.next_others = np.empty((capacity, 4), dtype=np.float32)



        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, others, next_others):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx]