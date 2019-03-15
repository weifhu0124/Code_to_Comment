import numpy as np
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch import optim
import torch.nn.functional as F


class Encoder(nn.Module):
    # word_size : the size of sbt vocabulary
    # emb_dim : the dimension to represent one vocabulary
    def __init__(self, word_size, emb_dim):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(word_size, emb_dim)
        self.lstm = nn.LSTM(word_size * emb_dim, word_size * emb_dim)


    def forward(self, input):
        # dimension: word_size x emb_dim -> word_size * emb_dim x 1
        output = self.embedding(input).view(1, 1, -1)

        # output.size = number of features
        output, hidden = self.lstm(output)

        # hidden state will be used to compute attention
        return output, hidden


