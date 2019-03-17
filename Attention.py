import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as function
import numpy as np
import torch.nn.init as torch_init

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_size
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):

        #  encoder_outputs:(seq_len, batch_size, hidden_size)
        #  hidden:(num_layers * num_directions, batch_size, hidden_size)

        max_len = encoder_outputs.size(0)
        h = hidden[-1].repeat(max_len, 1, 1)
        # (seq_len, batch_size, hidden_size)
        attn_energies = self.score(h, encoder_outputs)  # compute attention score
        return function.softmax(attn_energies, dim=1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):

        # (seq_len, batch_size, 2*hidden_size)-> (seq_len, batch_size, hidden_size)
        energy = function.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        return energy.squeeze(1)  # (batch_size, seq_len)
