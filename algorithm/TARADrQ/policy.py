import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils


class Encoder(nn.Module):

    def __init__(self, obs_shape, feature_dim=8):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 8
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, 2, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 2, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 2, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters,