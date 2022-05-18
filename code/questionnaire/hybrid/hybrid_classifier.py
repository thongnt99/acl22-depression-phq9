import torch
from torch import nn
from torch.nn import functional as F


class HybridClassifier(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pool="max"):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc1 = nn.Linear(embedding_dim, 1)
        self.fc2 = nn.Linear(5, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.convs.apply(self.init_weights)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
        self.pool = pool
        self.relu = nn.ReLU()

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.1)

    def forward(self, bert_output):
        ## convert to (batch_size, clhannel, seq_len, embedding_size)
        avg_emb = bert_output.mean(dim=1)
        bert_output = bert_output.unsqueeze(1)
        conved = [F.relu(conv(bert_output)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        topic = self.relu(self.fc1(avg_emb))
        output = cat*topic
        final_output = self.fc2(output)
        if final_output.size(1) == 1:
            return F.sigmoid(final_output), output
        else:
            return F.softmax(final_output, 1), output
