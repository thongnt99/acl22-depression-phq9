from torch import nn
from torch.nn import functional as F
from torch.nn import ModuleList
from transformers import BertConfig, BertLayer


class BertLinearClassifier(nn.Module):
    def __init__(self, bert_config_name, embedding_size, hidden_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.config = BertConfig.from_pretrained(bert_config_name)
        self.bert_layers = ModuleList([BertLayer(self.config) for _ in range(4)])
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights: Steal from thf"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, bert_output):
        # get [CLS] token only
        bert_output = (bert_output, )
        for idx in range(4):
            bert_output = self.bert_layers[idx](bert_output[0])
        batch_mean = bert_output[0].mean(dim=1)
        # batch_cls = bert_output[0][:, 0, :]
        output1 = self.fc1(batch_mean)
        input1 = F.relu(output1)
        output2 = self.fc2(input1)
        return F.softmax(output2, 1), output1
