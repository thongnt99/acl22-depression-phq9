import torch
import torch.nn as nn
from model.common.utils import batch_matmul_bias
from model.common.utils import batch_matmul
from model.common.utils import attention_mul


class AttentionWordRNN(nn.Module):

    def __init__(self, embed_size, word_gru_hidden, bidirectional=True):

        super(AttentionWordRNN, self).__init__()
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        if bidirectional:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 2 * word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))

        self.softmax_word = nn.Softmax(dim=0)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

    def forward(self, embed, state_word):
        self.word_gru.flatten_parameters()
        output_word, state_word = self.word_gru(embed, state_word)
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn)
        word_attn_vectors = attention_mul(output_word, word_attn_norm)
        return word_attn_vectors, state_word, word_attn_norm

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros(2, batch_size, self.word_gru_hidden)
        else:
            return torch.zeros(1, batch_size, self.word_gru_hidden)
