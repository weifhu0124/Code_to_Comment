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
	def __init__(self, input_size, word_size, emb_dim, hidden_size, device=None):
		super(Encoder, self).__init__()
		self.embedding = nn.Embedding(word_size, emb_dim)
		self.lstm = nn.LSTM(input_size=input_size * emb_dim, hidden_size=hidden_size)
		self.hidden = self._init_hidden(hidden_size, device)
		self.hidden_list = []
		self.hidden_list.append(self.hidden)

	# input is an index vector
	def forward(self, input):
		self.hidden = [h.detach() for h in self.hidden]
		# dimension: word_size x emb_dim -> word_size * emb_dim x 1
		output = self.embedding(input).view(1, 1, -1)

		# output.size = number of features
		output, self.hidden = self.lstm(output, self.hidden)
		self.hidden_list.append(self.hidden)

		# hidden state will be used to compute attention
		return output, self.hidden, self.hidden_list

	def _init_hidden(self, hidden_size, device):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		return [torch.zeros(1, 1, hidden_size).to(device),torch.zeros(1, 1, hidden_size).to(device)]



