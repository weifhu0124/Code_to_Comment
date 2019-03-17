import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Attention import Attention
import torch.nn.init as torch_init


class AttnDecoderRNN:
	def __init__(self, hidden_size, output_size, dropout_p=0.1):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		# word embedding
		# size of dictionary for embedding = output_size
		# size of embedding vector = hidden_size
		self.embedding = nn.Embedding(output_size, hidden_size)
		self.attn = Attention(hidden_size)
		# combine attention and inputs
		self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
		self.dropout = nn.Dropout(dropout_p)
		self.lstm = nn.LSTM(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		torch_init.xavier_normal_(self.out.weight)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)
		attn_weights = self.attn()
		# batch matrix multiplication
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

		# combine embedded input with attention
		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		# output = F.relu(output)
		output, hidden = self.lstm(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		return torch.zeros(1, 1, self.hidden_size, device=device)
