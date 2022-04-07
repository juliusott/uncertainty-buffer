import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 512)
        self._h2 = nn.Linear(512, 256)
        self._h3 = nn.Linear(256, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('tanh'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)*0.4

        return a

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 512)
        self._h2 = nn.Linear(512, 256)
        self._h3 = nn.Linear(256, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)

class MultiHeadCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, head_prob = 0.7, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.head_prob = head_prob
        self.n_heads = 10
        self.head_prob_dist = torch.distributions.bernoulli.Bernoulli(torch.tensor([head_prob for _ in range (self.n_heads)]))
        self.heads_mask = self.head_prob_dist.sample()

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)

        self.head0 = nn.Linear(n_features, n_output)
        self.head1 = nn.Linear(n_features, n_output)
        self.head2 = nn.Linear(n_features, n_output)
        self.head3 = nn.Linear(n_features, n_output)
        self.head4 = nn.Linear(n_features, n_output)
        self.head5 = nn.Linear(n_features, n_output)
        self.head6 = nn.Linear(n_features, n_output)
        self.head7 = nn.Linear(n_features, n_output)
        self.head8 = nn.Linear(n_features, n_output)
        self.head9 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.head0.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head1.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head2.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head3.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head4.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head5.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head6.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head7.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head8.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.head9.weight,
                                gain=nn.init.calculate_gain('linear'))

    def update_heads_mask(self):
        while self.heads_mask.sum() == 0:
            self.heads_mask = self.head_prob_dist.sample()

    def get_heads_mask(self):
        return self.heads_mask

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q0 = self.head0(features2)
        q1 = self.head1(features2)
        q2 = self.head2(features2)
        q3 = self.head3(features2)
        q4 = self.head4(features2)
        q5 = self.head5(features2)
        q6 = self.head6(features2)
        q7 = self.head7(features2)
        q8 = self.head8(features2)
        q9 = self.head9(features2)
        q = torch.cat((q0, q1, q2, q3, q4, q5, q6, q7 , q8, q9), dim=-1)
        return torch.squeeze(q)