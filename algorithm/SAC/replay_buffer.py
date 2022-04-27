import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device


        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.others = np.empty((capacity, 4), dtype=np.float32)
        self.next_others = np.empty((capacity, 4), d