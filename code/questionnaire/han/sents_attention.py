import torch
import torch.nn as nn
from model.common.utils import attention_mul
from model.common.utils import batch_matmul_bias
from model.common.utils import batch_matmul


class AttentionSentRNN(nn.Module):

    def __init__(self, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True, hidden_dim=5):
        super(AttentionSentRNN, self).__init__()
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        if bidirectional:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 2 * sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
            self.linear1 = nn.Linear(2 * sent_gru_hidden, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.linear1 = nn.Linear(sent_gru_hidden, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, n_classes)
        self.softmax_sent = nn.Softmax(dim=0)
        self.final_softmax = nn.Softmax(dim=1)
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1, 0.1)

    def forward(self, word_attention_vectors, state_sent):
        self.sent_gru.flatten_parameters()
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn)
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm)
        # final classifier
        output1 = self.linear1(sent_attn_vectors.squeeze(0))
        output2 = self.linear2(output1)
        return self.final_softmax(output2), state_sent, output1

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros(2, batch_size, self.sent_gru_hidden)
        else:
            return torch.zeros(1, batch_size, self.sent_gru_hidden)
