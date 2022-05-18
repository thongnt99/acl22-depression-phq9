from torch import nn
from torch.nn import functional as F


class LinearClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()

    def forward(self, bert_output):
        # get [CLS] token only
        batch_cls = bert_output.mean(dim=1)
        output1 = self.fc1(batch_cls)
        input1 = F.relu(output1)
        output2 = self.fc2(input1)
        return F.softmax(output2, 1), output1
