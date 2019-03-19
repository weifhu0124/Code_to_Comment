# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import SBT_encode
import pickle
import numpy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden, encoder_output):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden, encoder_output

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)



######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_len = 2000

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								 encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(
			input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden


	# Teacher forcing: Feed the target as the next input
	for di in range(target_length):
		decoder_output, decoder_hidden, decoder_attention = decoder(
			decoder_input, decoder_hidden, encoder_outputs)
		loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
		decoder_input = target_tensor[di]  # Teacher forcing

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def preprocessing(file_name, type, comment_dict=None):
	# load data
	with open(file_name, 'rb') as f:
		data = pickle.load(f)

	comment = []
	code = []
	for i in range(len(data)):
		temp_comment, temp_code = data[i]
		comment.append(temp_comment)
		code.append(temp_code)

	pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
	# only when training
	if comment_dict == None:
		commment_wordlist = []
		for i in range(len(comment)):
			temp_list = re.split(pattern, comment[i])
			for x in temp_list:
				commment_wordlist.append(x)
		temp_dict = {}
		for word in commment_wordlist:
			temp_dict[word] = commment_wordlist.count(word)
		temp_wordlist = sorted(temp_dict.items(), key=lambda kv: (-kv[1], kv[0]))[:3000]
		commment_wordlist = [temp_wordlist[i][0] for i in range(len(temp_wordlist))]

		comment_dict = dict(zip(commment_wordlist, range(2, len(commment_wordlist)+2)))
		comment_dict[0] = 'UNK'
		comment_dict[1] = 'EOS'
		# save dictionary
		with open('data/comment_dict.pkl', 'wb') as pfile:
			pickle.dump(comment_dict, pfile)

	encoder = SBT_encode.Encoder()

	code_in_num = []
	comment_in_num = []

	for i in range(len(code)):
		code_in_num.append(encoder.encode(code[i]))
	code_in_num.append(6799)

	for i in range(len(comment)):
		split_list = re.split(pattern, comment[i])
		temp_list = []
		for x in split_list:
			if x in comment_dict:
				temp_list.append(comment_dict[x])
			else:
				temp_list.append(0)
		temp_list.append(1)
		comment_in_num.append(temp_list)
	with open('data/' + type + '_code_in_num.pkl', 'wb') as pfile:
		pickle.dump(code_in_num, pfile)
	with open('data/' + type + '_comment_in_num.pkl', 'wb') as pfile:
		pickle.dump(comment_in_num, pfile)
	return code_in_num, comment_in_num, comment_dict

# could also be use to test
def validate_model(encoder, decoder, criterion, loader, device=None, verbose=False):
	val_loss = 0
	with torch.no_grad():
		indices = np.arange(len(loader[1]))
		np.random.shuffle(indices)
		indices = indices[0:500]
		for i in range(len(indices)):
			encoder_hidden = encoder.initHidden()
			# Put the minibatch data in CUDA Tensors and run on the GPU if supported
			inputs, targets = loader[0][indices[i]], loader[1][indices[i]]
			input_tensor = torch.LongTensor(inputs).to(device)
			target_tensor = torch.LongTensor(targets).to(device)
			input_length = input_tensor.size(0)
			target_length = target_tensor.size(0)

			encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

			loss = 0

			for ei in range(input_length):
				encoder_output, encoder_hidden = encoder(
					input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] = encoder_output[0, 0]

			decoder_input = torch.tensor([[SOS_token]], device=device)

			decoder_hidden = encoder_hidden

			# Teacher forcing: Feed the target as the next input
			for di in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_outputs)
				loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
				decoder_input = target_tensor[di]  # Teacher forcing
			val_loss += loss.item() / target_length
		print('Validation Loss: ', val_loss / len(indices))
	return val_loss /len(indices)

def trainIters(validate_every=5000, learning_rate=0.05):
	epochs = 50
	plot_train_losses = []
	plot_val_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every
	hidden_size = 256
	print('------- Hypers --------\n'
		  '- epochs: %i\n'
		  '- learning rate: %g\n'
		  '- hidden size: %i\n'
		  '----------------'
		  '' % (epochs, learning_rate, hidden_size))

	criterion = nn.NLLLoss()
	# train_code_in_num, train_comment_in_num, train_comment_dict = preprocessing('data/train.pkl', 'train')
	# val_code_in_num, val_comment_in_num, train_comment_dict = preprocessing('data/valid.pkl', 'val', train_comment_dict)
	# test_code_in_num, test_comment_in_num, train_comment_dict = preprocessing('data/test.pkl', 'test', train_comment_dict)
	train_word_size_encoder = 6800
	train_word_size_decoder = 3002
	with open('data/train_code_in_num.pkl', 'rb') as pfile:
		train_code_in_num = pickle.load(pfile)
	with open('data/train_comment_in_num.pkl', 'rb') as pfile:
		train_comment_in_num = pickle.load(pfile)
	with open('data/val_code_in_num.pkl', 'rb') as pfile:
		val_code_in_num = pickle.load(pfile)
	with open('data/val_comment_in_num.pkl', 'rb') as pfile:
		val_comment_in_num = pickle.load(pfile)
	with open('data/test_code_in_num.pkl', 'rb') as pfile:
		test_code_in_num = pickle.load(pfile)
	with open('data/test_comment_in_num.pkl', 'rb') as pfile:
		test_comment_in_num = pickle.load(pfile)
	print('Data Loaded')

	encoder = EncoderRNN(train_word_size_encoder, hidden_size).to(device)
	decoder = DecoderRNN(hidden_size, train_word_size_decoder).to(device)
	#decoder = AttnDecoderRNN(hidden_size, train_word_size_decoder, dropout_p=0.1).to(device)
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

	dataloaders = {}
	dataloaders['train'] = (train_code_in_num, train_comment_in_num)
	dataloaders['val'] = (val_code_in_num, val_comment_in_num)
	dataloaders['test'] = (test_code_in_num, test_comment_in_num)
	counts = []
	count = 1

	for eps in range(0, epochs):
		for iter in range(0, len(dataloaders['train'][1])):
			inputs, targets = dataloaders['train'][0][iter], dataloaders['train'][1][iter]
			inputs = torch.LongTensor(inputs)
			targets = torch.LongTensor(targets)
			inputs, targets = inputs.to(device), targets.to(device)

			loss = train(inputs, targets, encoder,
							 decoder, encoder_optimizer, decoder_optimizer, criterion)
			plot_loss_total += loss
			print(iter, loss)
			if iter % validate_every == 0:
				counts.append(count)
				count += 1
				plot_loss_avg = plot_loss_total / validate_every
				plot_train_losses.append(plot_loss_avg)
				val_loss = validate_model(encoder, decoder, criterion, dataloaders['val'], device=device)
				plot_val_losses.append(val_loss)
				plot_loss_total = 0

	showPlot(count, plot_train_losses, plot_val_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(iter, train_loss, val_loss):
	plt.figure()
	plt.plot(iter, train_loss, '.-', label='train')
	plt.plot(iter, val_loss, '.-', label='val')
	fontsize = 12
	plt.legend(fontsize=fontsize)
	plt.xlabel('epoch', fontsize=fontsize)
	plt.ylabel('loss', fontsize=fontsize)
	plt.savefig('loss.png', fontsize=fontsize)


trainIters()

######################################################################
#

# evaluateRandomly(encoder1, attn_decoder1)
#
#
# ######################################################################
# # Visualizing Attention
# # ---------------------
# #
# # A useful property of the attention mechanism is its highly interpretable
# # outputs. Because it is used to weight specific encoder outputs of the
# # input sequence, we can imagine looking where the network is focused most
# # at each time step.
# #
# # You could simply run ``plt.matshow(attentions)`` to see attention output
# # displayed as a matrix, with the columns being input steps and rows being
# # output steps:
# #
#
# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())
#
#
# ######################################################################
# # For a better viewing experience we will do the extra work of adding axes
# # and labels:
# #
#
# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
