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
            nn.Conv2d(self.num_filters, self.num_filters, 2, stride=2)
        ])


        self.outputs = dict()

    def forward_conv(self, obs):
        obs = torch.unsqueeze(obs, dim=1)
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = nn.functional.adaptive_avg_pool2d(h, (1,1))
        out = torch.squeeze(out, 3)
        out = torch.squeeze(out, 2)


        return out

    def copy_conv_weights_from(self, source):
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])



class Actor(nn.Module):

    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds, feature_dim=8):
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        self.log_std_bounds = log_std_bounds

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2 * action_shape[0])
        )

        self.lstm = nn.LSTMCell(8, 8)
        self.ta = nn.Linear(2 * 8, 2)
        self.outputs = dict()
        self.apply(utils.weight_init)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, obs, others, detach_encoder=False):

        obs0 = self.encoder(obs[:, 0, :, :], detach=detach_encoder)
        obs1 = self.encoder(obs[:, 1, :, :], detach=detach_encoder)
        obs2 = self.encoder(obs[:, 2, :, :], detach=detach_encoder)
        obs3 = self.encoder(obs[:, 3, :, :], detach=detach_encoder)

        xs = []
        xs.append(obs0)
        xs.append(obs1)
        xs.append(obs2)
        xs.append(obs3)
        ht = torch.zeros(obs.size()[0], 8).to("cuda")
        ct = torch.zeros(obs.size()[0], 8).to("cuda")
        h_list = []

        for x in xs:
            ht, ct = self.lstm(x, (ht, ct))
            h_list.append(ht)

        total_ht = h_list[0]
        for i in range(1, len(h_list)):
            total_ht = torch.cat((total_ht, h_list[1]), 1)
        beta_t = self.relu(self.ta(total_ht))
        beta_t = self.softmax(beta_t)
        out = torch.zeros(obs.size()[0], 8).to("cuda")
        for i in range(len(h_list)):
            out = out + h_list[i] * beta_t[:, i].reshape(obs.size()[0], 1)

        obs = out

        obs = torch.cat([obs, others], dim=1)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()


        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):

    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth, feature_dim=8):
        super().__init__()

        self.encoder = Encoder(obs_shape, feature_dim)

        self.Q1 = nn.Sequential(
            nn.Linear(self.encoder.feature_d