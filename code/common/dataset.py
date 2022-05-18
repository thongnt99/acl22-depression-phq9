from torch.utils.data import Dataset
import torch
from model.common.utils import batch_tokenize, get_batch_bert_embedding
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import math

BATCH_SIZE = 16

class DepressionTextDataset(Dataset):
    """Dataset of users with raw text posts"""
    def __init__(self, path):
        super(RSDD, self).__init__()
        self.users = []
        with open(path, "r") as f:
            for line in f:
                try:
                    user = json.loads(line)[0]
                except json.JSONDecoderError:
                    continue
                if user["label"] == "depression" or user["label"] == "control":
                    self.users.append(user)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        return self.users[item]

# default collate function does not support dictionary data
def collate_fn(batch):
    return batch



class SymptomDataset(Dataset):
    """Data set for mental-health symptoms detection: each instance is a text post"""

    def __init__(self, path, bert_tokenizer, bert_model, remove_method=None):
        super().__init__()
        self.texts = []
        self.labels = []
        self.bert_embeddings = []
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        selected_methods = ["mental_health", "smhd", "keyword", "aug_negation", "pronoun", "sentiment", "other"]
        if remove_method != "all" and remove_method is not None:
            action = remove_method[0]
            method = remove_method[1:]
            if action == "-":
                selected_methods.remove(method)
            else:
                selected_methods.append(method)
        with open(path, "r") as f:
            for line in f:
                try:
                    post = json.loads(line.strip())
                    if remove_method is None or ("method" in post and post["method"] in selected_methods):
                        self.texts.append(post["text"])
                        self.labels.append(int(post["label"]))
                except json.JSONDecodeError:
                    continue
        self.labels = torch.tensor(self.labels)
        self.convert_to_bert_embedding()

    def convert_to_bert_embedding(self):
        num_batch = math.ceil(len(self.texts) / BATCH_SIZE)
        for i in range(num_batch):
            texts = self.texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            batch_x = batch_tokenize(self.bert_tokenizer, texts)
            batch_x = get_batch_bert_embedding(self.bert_model, batch_x)
            self.bert_embeddings.append(batch_x.detach().to("cpu"))
        self.bert_embeddings = torch.cat(self.bert_embeddings, 0)
        assert self.bert_embeddings.size(0) == len(self.texts)

    def get_class_weights(self):
        class_1 = self.labels.sum()
        class_0 = (1-self.labels).sum()
        eps = 1e-5
        total = class_0 + class_1 + eps
        weights = [(class_1+eps)/total, (class_0+eps)/total]
        print("Class weight: {} {}".format(weights[0], weights[1]))
        return torch.tensor(weights)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bert_embeddings[idx], self.labels[idx]


class DepressionEmbeddingDataset(Dataset):
    """Dataset of users with post embeddings"""

    def __init__(self, path, post_time_path=None, m_format="pickle", data_field="input", threshold_to_remove=1.0, question_to_remove=-1,
                 donwsample=None):
        super(DepressionEmbeddingDataset, self).__init__()
        self.inputs = []
        self.labels = []
        if data_field == "agg2week":
            self.post_times = json.load(open(post_time_path))
        if m_format == "pickle":
            self.load_from_pickle(path, data_field, threshold_to_remove, question_to_remove)
        elif m_format == "json":
            self.load_from_json(path, data_field)
        else:
            raise ValueError("This format is not supported")
        if donwsample == True:
            np.random.seed(10)
            neg_index = [i for i in range(len(self.labels)) if self.labels[i] == 0]
            pos_index = [i for i in range(len(self.labels)) if self.labels[i] == 1]
            neg_index = np.random.permutation(neg_index)
            neg_index = neg_index[:len(pos_index)].tolist()
            selected_index = neg_index + pos_index
            self.inputs = [self.inputs[idx] for idx in selected_index]
            self.labels = [self.labels[idx] for idx in selected_index]

    def load_from_pickle(self, path, data_field, threshold_to_remove, question_to_remove):
        users = pickle.load(open(path, "rb"))
        for idx, user in enumerate(users):
            if data_field == "input":
                inp = user["input"]
            elif data_field == "state":
                inp = user["states"].to("cpu")
                inp = inp[:400, :]
                if question_to_remove >= 0:
                    inp[:, question_to_remove*5 : (question_to_remove+1)*5] = 0
            elif data_field == "post":
                inp = user["posts"]
                if inp.dim() == 1:
                    inp = inp.unsqueeze(0)
                inp = inp[:400, :]
                # scores = user["scores"]
                # idx_to_remove = (scores[:, question_to_remove] >= threshold_to_remove)
                # idx_to_remove = idx_to_remove[: inp.size(0)]
                # inp[idx_to_remove, :] = float("-inf")
                # assert inp[idx_to_remove, :].sum() == float("-inf"), \
                #     "actual: {}, expected: {}".format(inp[idx_to_remove, :].sum(), float("-inf"))
            elif data_field == "score":
                inp = user["scores"][:400, :]
                if question_to_remove >= 0:
                    inp[:, question_to_remove] = 0
            elif data_field == "agg2week":
                # inp = self.aggregate_score(user["scores"], self.post_times[idx])
                n = min(user["scores"].size(0), len(self.post_times[idx]))
                t = torch.tensor(self.post_times[idx][:n]).unsqueeze(1)
                inp = torch.cat([user["scores"][:n], t], dim=1)
                inp = self.aggregate_score(inp)
            elif data_field == "sum_score":
                inp = user["scores"].sum(0)
            else:
                raise ValueError("data_field {} does not exist".format(data_field))
            self.inputs.append(inp)
            self.labels.append(user["label"])

    @staticmethod
    def aggregate_score(inp, time_window_in_days=14):
        time_window_in_seconds = time_window_in_days*24*60*60
        start_idx = 0
        slides = []
        max_length = 0
        for i in range(inp.size(0)):
            if inp[i][-1] - inp[start_idx][-1] >= time_window_in_seconds:
                slides.append(inp[start_idx:i])
                max_length = max(max_length, i-start_idx)
                start_idx = i
        slides.append(inp[start_idx:i])
        max_length = max(max_length, i - start_idx)
        # pad
        padded = torch.zeros((len(slides), max_length, inp.size(1)))
        for i, s in enumerate(slides):
            padded[i, :s.size(0)] = s
        return padded

    def load_from_json(self, path, data_field):
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                if line.strip() == "":
                    continue
                try:
                    user = json.loads(line)
                    if data_field == "input":
                        inp = torch.tensor(user["input"])
                    elif data_field == "post":
                        inp = torch.tensor(user["posts"][:400,:])
                        if inp.dim() == 1:
                            inp = inp.unsqueeze(0)
                    elif data_field == "score":
                        inp = torch.tensor(user["scores"])[:400,:]
                    elif data_field == "sum_score":
                        inp = torch.tensor(user["scores"])[:400,:].sum(0)
                    elif data_field == "agg2week":
                        inp = self.aggregate_score(torch.tensor(user["scores"]), self.post_times[idx])
                    else:
                        raise ValueError("Data field {} does not exist".format(data_field))
                    self.inputs.append(inp)
                    self.labels.append(torch.tensor(user["label"]))
                except Exception as e:
                    print(e)
                    continue

    def sample_balanced_batch(self, batch_size):
        pos_idx = np.random.permutation([i for i in range(len(self.labels)) if self.labels[i] == 1])
        neg_idx = np.random.permutation([i for i in range(len(self.labels)) if self.labels[i] == 0])
        batch = []
        for i in range(batch_size//2):
            batch.append((self.inputs[pos_idx[i]], self.labels[pos_idx[i]]))
            batch.append((self.inputs[neg_idx[i]], self.labels[neg_idx[i]]))
        return batch

    def get_weights(self):
        unique_labels, freqs = np.unique(torch.tensor(self.labels).tolist(), return_counts=True)
        probs = 1 / freqs
        for p in probs:
            assert (0 < p < 1)
        label2weight = dict(zip(unique_labels, probs))
        weights = [label2weight[int(lab)] for lab in self.labels]
        return weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]