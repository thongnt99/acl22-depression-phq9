import sys
from scipy._lib.decorator import __init__
sys.path.append(sys.path[0] + '/..')
from transformers import AutoModel, AutoTokenizer
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from model.common.dataset import SymptomDataset
from torch.optim.lr_scheduler import OneCycleLR
from model.common.utils import get_bert_details_from_alias
from model.common.functional import get_classifier_from_name
from model.common.functional import train, test

parser = argparse.ArgumentParser(description="Training model using bert")
parser.add_argument("--train", type=str, help="path to a train file (jsonl)")
parser.add_argument("--dev", type=str, help="path to a dev file (jsonl)")
parser.add_argument("--test", type=str, help="path to a test file (jsonl)")
parser.add_argument("--save", type=str, help="path to save model")
parser.add_argument("--epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--es_patience", type=int, default=1, help="patience before stop training")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
parser.add_argument("--l2", type=float, default=0.001, help="L2 regularizing weight")
parser.add_argument("--bert", type=str, default=64, help="batch size")
parser.add_argument("--model", type=str, default="cnn", help="Specify model: cnn, linear, han, hybrid")
parser.add_argument("--hidden_dim", type=int, default=50, help="dimension of the hidden layer")
parser.add_argument("--sig", type=str, default="all", help="signature of the model: differentiate with other settings")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Loading bert model")
    bert_size, bert_path = get_bert_details_from_alias(args.bert)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    device_list = list(range(torch.cuda.device_count()))
    bert_model = nn.DataParallel(AutoModel.from_pretrained(bert_path).to(device), device_ids=device_list)
    mh_classifier = get_classifier_from_name(bert_path, args.model, bert_size, hidden_dim=args.hidden_dim)
    mh_classifier = nn.DataParallel(mh_classifier.to(device), device_ids=device_list)
    if args.model == "hybrid" or args.model == "cnn":
        model_path = os.path.join(args.save,
                                  "{}_{}_bert_{}_l2_{}_{}.md".format(args.model, args.hidden_dim, args.bert, args.l2, args.sig))
    else:
        model_path = os.path.join(args.save, "{}_{}_bert_{}_{}.md".format(args.model, args.hidden_dim, args.bert, args.sig))

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5])).to(device)
    if (args.train is not None) and (args.dev is not None):
        print("Loading train data")
        train_data = SymptomDataset(args.train, bert_tokenizer, bert_model)
        train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        dev_data = SymptomDataset(args.dev, bert_tokenizer, bert_model)
        dev_dl = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(mh_classifier.parameters(), lr=args.lr, eps=1e-8, weight_decay=0.1)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dl), epochs=args.epoch)
        train(mh_classifier, train_dl, dev_dl, optimizer, criterion, scheduler, args.epoch,
              args.es_patience, model_path, device=device, one_cycle_lr=True, regularize=False, regularize_weight=args.l2)

    test_data = SymptomDataset(args.test, bert_tokenizer, bert_model)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    mh_classifier.load_state_dict(torch.load(model_path))
    test(mh_classifier, test_dl, criterion, device=device)
