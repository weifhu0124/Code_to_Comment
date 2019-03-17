import numpy as np
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch import optim
import torch.nn.functional as F

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		# hidden size is the same as encoder
		self.hidden_size = hidden_size

		# word embedding
		# size of dictionary for embedding = output_size
		# size of embedding vector = hidden_size
		self.embedding = nn.Embedding(output_size, hidden_size)
		# LSTM unit
		# output of embedding would be = hidden_size
		self.lstm = nn.LSTM(output_size*hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		torch_init.xavier_normal_(self.out.weight)
		#self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output, hidden = self.lstm(output, hidden)
		output = self.out(output[0])
		# output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		return torch.zeros(1, 1, self.hidden_size, device=device)
