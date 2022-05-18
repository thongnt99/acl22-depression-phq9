import torch
from torch import nn
from model.questionnaire.han.sents_attention import AttentionSentRNN
from model.questionnaire.han.words_attention import AttentionWordRNN
from model.common.utils import split_and_pad_batch
TOKEN_PER_SENTENCE = 15

class HANClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_dim=5):
        super(HANClassifier, self).__init__()
        self.piece_attention = AttentionWordRNN(embed_size=embedding_size,
                                                word_gru_hidden=100, bidirectional=True)
        self.part_attention = AttentionSentRNN(sent_gru_hidden=100, word_gru_hidden=100,
                                               n_classes=2, bidirectional=True, hidden_dim=hidden_dim)

    def forward(self, x_batch):
        x_batch = split_and_pad_batch(x_batch, TOKEN_PER_SENTENCE)
        _state_word = self.piece_attention.init_hidden(x_batch.size()[0]).to(x_batch.device)
        _state_sent = self.part_attention.init_hidden(x_batch.size()[0]).to(x_batch.device)
        batch_size, max_sent, max_tokens, emb_size = x_batch.size()
        s = []
        for i in range(max_sent):
            _s, state_word, _ = self.piece_attention(x_batch[:, i, :, :].transpose(0, 1), _state_word)
            s.append(_s)
        s = torch.cat(s, 0)
        y_pred, state_sent, last_hidden = self.part_attention(s, _state_sent)
        return y_pred, last_hidden
