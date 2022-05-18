from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.functional import softmax, sigmoid



class LinearDepressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearDepressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 2)
        self.softmax = nn.Softmax(1)
        # self.init_weights(self.linear)
        # self.lost_criterion = nn.CrossEntropyLoss()

    @staticmethod
    def init_weights(layer):
        for name, w in layer.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(w)
            else:
                nn.init.uniform_(w)

    def forward(self, x_batch):
        linear_out = self.linear(x_batch)
        return self.softmax(linear_out), None

class CNNDepressionModel(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pool="max"):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        if pool == "k-max":
            self.fc = nn.Linear(len(filter_sizes)*n_filters*5, output_dim)
        elif pool == "mix":
            self.fc = nn.Linear(len(filter_sizes)*n_filters*10, output_dim)
        else:
            self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.convs.apply(self.init_weights)
        self.init_weights(self.fc)
        self.pool = pool

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.1)

    def forward(self, bert_output):
        ## convert to (batch_size, clhannel, seq_len, embedding_size)
        bert_output = bert_output.unsqueeze(1)
        conved = [F.relu(conv(bert_output)).squeeze(3) for conv in self.convs]
        if self.pool == "max":
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        elif self.pool == "k-max":
            bs = bert_output.size(0)
            pooled = [conv.topk(5, dim=2)[0].view(bs, -1) for conv in conved]
        elif self.pool == "mix":
            bs = bert_output.size(0)
            pooled = [torch.cat([conv.topk(5, dim=2)[0].view(bs, -1),
                                     conv.topk(5, dim=2, largest=False)[0].view(bs, -1)], dim=1) for conv in conved]
        elif self.pool == "avg":
            pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        else:
            raise ValueError("This kernel is currently not supported.")

        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        output = self.fc(cat)
        if output.size(1) == 1:
            return F.sigmoid(output), cat
        else:
            return F.softmax(output, 1) , cat