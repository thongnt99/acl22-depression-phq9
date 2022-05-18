import sys

sys.path.append(sys.path[0] + '/..')
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from random import randint
from torch.optim.lr_scheduler import OneCycleLR
from model.common.dataset import DepressionEmbeddingDataset
from model.common.utils import get_last_hidden_size
from model.depression.depression_model import LinearDepressionModel, CNNDepressionModel
from model.common.functional import train, test

symptom_names = ["anhedonia", "concentration", "eating", "fatigue", "mood", "psychomotor", "self-esteem", "self-harm",
                 "sleep"]
parser = argparse.ArgumentParser("Extract symptom from text")
parser.add_argument("--train", type=str, help="Training file")
parser.add_argument("--dev", type=str, help="Development file")
parser.add_argument("--test", type=str, help="Testing file")
parser.add_argument("--train_time", type=str, help="Posting time in training file")
parser.add_argument("--dev_time", type=str, help="Posting time in development file")
parser.add_argument("--test_time", type=str, help="Posting time in testing file")
parser.add_argument("--epochs", type=int, default=60, help="Number of epoch")
parser.add_argument("--es_patience", type=int, default=1, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=16, help="user batch size")
parser.add_argument("--lr", type=float, default=0.001, help="user batch size")
parser.add_argument("--save", type=str, help="Save")
parser.add_argument("--model", type=str, help="model version (han,cnn, linear)")
parser.add_argument("--resume", type=str, help="Resume")
parser.add_argument("--fold", type=int, help="which fold to train")
parser.add_argument("--only_evaluate", default=False, action="store_true")
parser.add_argument("--rounds", type=int, default=10, help="number of training rounds")
parser.add_argument("--top_model", type=str, default="linear", help="top model")
parser.add_argument("--dim", type=int, default=9, help="dimension of the input")
parser.add_argument("--data_field", type=str, default="post", help="which data field to process")
parser.add_argument("--data_format", type=str, default="pickle", help="pickle or json")
parser.add_argument("--threshold", type=float, default=0.8, help="decision threshold")
parser.add_argument("--pooling", type=str, default="k-max", help="CNN pooling strategy")


def get_label(data, idx):
    return data[idx][1]


def collate_fn(batch):
    users = [item[0] for item in batch]  # users x posts
    labels = [item[1] for item in batch]
    max_post = max(max([u.size(0) for u in users]), 10)
    x_batch = torch.zeros(len(users), max_post, args.dim)
    for idx in range(len(users)):
        x_batch[idx, :users[idx].size(0), :users[idx].size(1)] = users[idx]
    y_batch = torch.tensor(labels)
    return x_batch, y_batch


def collate_fn_window(batch):
    users = [item[0] for item in batch]  # users x posts
    labels = [item[1] for item in batch]
    max_slides = max([u.size(0) for u in users])
    max_post = max([u.size(1) for u in users])
    x_batch = torch.zeros(len(users), max_slides, max_post, users[0].size(2))
    for i in range(len(users)):
        x_batch[i, :users[i].size(0), :users[i].size(1)] = users[i]
    y_batch = torch.tensor(labels)
    return x_batch, y_batch


class BalancedBatchingDataloader:
    def __init__(self, dataset, num_batches, b_s, collate_fn):
        self.num_batches = num_batches
        self.collate_fn = collate_fn
        self.dataset = dataset
        self.batch_size = b_s

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.collate_fn(self.dataset.sample_balanced_batch(self.batch_size))

    def __len__(self):
        return self.num_batches


def instantiate_classifier(top_model, dim, config_name="bert-base-uncased"):
    if top_model == "linear":
        classifier = LinearDepressionModel(dim)
    elif top_model == "cnn":
        classifier = CNNDepressionModel(embedding_dim=dim, n_filters=50, filter_sizes=[2, 3, 4, 5, 6],
                                   output_dim=2, dropout=0.5, pool=args.pooling)
    else:
        raise ValueError("The model {} is not implemented".format(top_model))
    return classifier


def train_with_seed(m_seed, top_model="cnn", dim=9, threshold=0.8):
    torch.manual_seed(m_seed)
    np.random.seed(m_seed)
    classifier = instantiate_classifier(top_model, dim)
    classifier.to(device)
    m_depression_model = nn.DataParallel(classifier,
                                         device_ids=device_list)
    optimizer = torch.optim.Adam(m_depression_model.parameters(), lr=args.lr, eps=1e-8)
    # scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs)
    if args.resume is not None:
        print("Loading depression model from {}".format(args.resume))
        m_depression_model.load_state_dict(torch.load(args.resume))
    saving_path = os.path.join(args.save, "depression_model_0_0_base_{}_seed_{}.md".format(args.model, m_seed))
    dev_f1 = train(m_depression_model, train_dataloader, dev_dataloader, optimizer, criterion, scheduler,
                   args.epochs, args.es_patience, saving_path, device=device, threshold=threshold,
                   one_cycle_lr=True)
    return dev_f1, saving_path


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0")
    device_list = [0]
    input_dim = len(symptom_names) * get_last_hidden_size(args.model)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.8])).to(device)
    # criterion = f1_loss
    # depression_model = nn.DataParallel(DepressionModelLSTM(1024, 100).to(device), device_ids=device_list)

    selected_collate_fn = collate_fn if args.data_field != "agg2week" else collate_fn_window
    if not args.only_evaluate:
        print("Loading data from {}".format(args.train))
        train_dataset = DepressionEmbeddingDataset(args.train, post_time_path=args.train_time, data_field=args.data_field,
                                             m_format=args.data_format)
        # sample_weights = train_dataset.get_weights()
        # imbalance_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=imbalance_sampler, num_workers=1,
        #                           collate_fn=collate_fn)
        # under sampling
        # train_dataset.under_sample_class_0()
        # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
        #                           collate_fn=collate_fn)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=1, collate_fn=selected_collate_fn)  # if args.top_model == "cnn" else None)
        print("Print sample using imbalanced sampler")
        train_iter_imbalanced = iter(train_dataloader)
        for _ in range(5):
            posts, targets = next(train_iter_imbalanced)
            print(posts.size())
            print("Classes {}, counts: {}".format(*np.unique(targets.numpy(), return_counts=True)))
        print("Loading data from {}".format(args.dev))
        dev_dataset = DepressionEmbeddingDataset(args.dev, post_time_path=args.dev_time, data_field=args.data_field,
                                           m_format=args.data_format)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, collate_fn=selected_collate_fn)  # if args.top_model == "cnn" else None)
        # # smallest_loss = 100
        # best_f1 = -1
        # best_path = None
        # best_seed = -1
        # best_threshold = -1
        this_seed = randint(0, 12345)
        for idx in range(1,10):
            threshold = idx*0.1
            print("=====================================")
            print("Training round {} with seed {} threshod {}".format(idx, this_seed, threshold))
            f1, path = train_with_seed(this_seed, top_model=args.top_model, dim=args.dim, threshold=threshold)
            # if smallest_loss > loss:
            if best_f1 <= f1:
                # smallest_loss = loss
                best_f1 = f1
                best_path = path
                print("Improvement at round {}, saving model to {}, best_f1 {}".format(idx, best_path, best_f1))
                best_threshold = threshold
    if args.only_evaluate:
        best_threshold = args.threshold
        best_path = args.resume
    clsfier = instantiate_classifier(args.top_model, args.dim, config_name="google/bert_uncased_L-12_H-768_A-12")
    clsfier.to(device)
    depression_model = nn.DataParallel(clsfier, device_ids=device_list)
    print("Loading the best depression model from {}".format(best_path))
    depression_model.load_state_dict(torch.load(best_path))
    print("Loading data from {}".format(args.test))
    test_dataset = DepressionEmbeddingDataset(args.test, post_time_path=args.test_time, data_field=args.data_field,
                                        m_format=args.data_format)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, collate_fn=selected_collate_fn)  # if args.top_model == "cnn" else None)
    test(depression_model, test_dataloader, criterion, device=device, threshold=best_threshold)
