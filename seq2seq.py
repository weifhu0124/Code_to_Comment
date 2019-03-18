import torch
import torch.nn as nn
from Encoder import Encoder
from DecoderRNN import DecoderRNN
from AttnDecoderRNN import AttnDecoderRNN


class seq2seq(nn.Module):
    # model = seq2seq(word_size_encoder, emb_dim, hidden_size, word_size_decoder, MAX_LEN)

    def __init__(self, word_size_encoder, emb_dim, hidden_size, word_size_decoder, max_length,criterion):
        super(seq2seq, self).__init__()
        self.max_length = max_length
        self.criterion = criterion
        self.encoder = Encoder(word_size_encoder, emb_dim, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, word_size_decoder, max_length)

    # input_encoder : vector of index
    def forward(self, input_encoder, input_decoder, teacher_forcing = True):
        loss = 0

        input_length = len(input_encoder)
        output_length = len(input_decoder)
        # encoder_outputs = torch.zeros(self.max_length, self.encoder.emb_dim)
        encoder_hidden = torch.zeros(self.max_length, self.encoder.hidden_size)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_encoder[ei])
            encoder_hidden[ei] = encoder_hidden[0, 0]

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[0]])
        decoder_outputs = torch.zeros(self.max_length, self.decoder.hidden_size)

        if teacher_forcing:

            for di in range(output_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, input_decoder[di])
                decoder_input = input_decoder[di]
                decoder_outputs[di] = decoder_output[0, 0]
                  # Teacher forcing
        else:
            # not teacher forcing
            decoder_input = torch.tensor([[0]])
            for di in range(output_length):

                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_outputs[di] = decoder_output[0, 0]
                loss += self.criterion(decoder_output, input_decoder[di])

        loss.backward()

        return loss, decoder_outputs