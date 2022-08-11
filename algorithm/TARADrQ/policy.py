import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils


class Encoder(nn.Module):

    def __init__(self, obs_shape, feature_dim=8