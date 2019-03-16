import torch
import torch.nn as nn
from Encoder import Encoder
from AnnDecoderRNN import AnnDecoderRNN


class seq2seq(nn.Module):
    def __init__(self, word_size_encoder, emb_dim, hidden_size, word_size_decoder):
        super(seq2seq, self).__init__()
        self.encoder = Encoder(word_size_encoder, emb_dim, hidden_size)
        self.decoder = AnnDecoderRNN(hidden_size, word_size_decoder)

    # input_encoder : vector of index
    def forward(self, input_encoder, input_decoder):
        output, hidden = self.encoder(input_encoder)
        output = self.decoder(input_decoder, hidden, hidden)
        return output
