import sys

sys.path.append(sys.path[0] + '/..')
import json
import argparse
import torch
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from model.common.dataset import DepressionTextDataset
from model.common.dataset import collate_fn
from model.common.utils import get_bert_details_from_alias
from model.common.functional import get_classifier_from_name
from model.questionnaire.questionnaire_model import QuestionnaireModel
from model.common.utils import get_last_hidden_size

NUM_EPOCHS = 60
TOKEN_PER_SENTENCE = 15
POSTS_PER_USER = 400
EMBEDDING_SIZE = 128
USER_BATCH_SIZE = 16
POST_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
ES_PATIENCE = 3
NUMBER_OF_SYMPTOMS = 9

symptom_names = ["anhedonia", "concentration", "eating", "fatigue", "mood", "psychomotor", "self-esteem", "self-harm", "sleep"]
parser = argparse.ArgumentParser("Extract symptom from text")
parser.add_argument("--input", type=str, help="Training file")
parser.add_argument("--bert", type=str, help="Bert version")
parser.add_argument("--model", type=str, help="Model version")
parser.add_argument("--output", type=str, help="Development file")
parser.add_argument("--hidden_dim", type=int, default=5, help="dimension of last hidden layer of questionnaire model")
parser.add_argument("--l2", type=float, default=None, help="which l2 weight used")
parser.add_argument("--log", type=str, default="log.txt", help="Log file")
args = parser.parse_args()

if __name__ == "__main__":
    device_list = list(range(torch.cuda.device_count()))
    print("Loading bert {} model".format(args.bert))
    bert_size, bert_path = get_bert_details_from_alias(args.bert)
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = nn.DataParallel(AutoModel.from_pretrained(bert_path).to("cuda:0"), device_ids=device_list)
    bert_model.eval()
    print("Loading symptom model")
    model_list = []
    for idx in range(len(symptom_names)):
        if args.l2 == None:
            path = os.path.join(
                *["models", symptom_names[idx], "{}_{}_bert_{}_all.md".format(args.model, args.hidden_dim, args.bert)])
        else:
            path = os.path.join(
                *["models", symptom_names[idx], "{}_{}_bert_{}_l2_{}_all.md".format(args.model, args.hidden_dim,
                                                                                 args.bert, args.l2)])
        print("Loading {} model from {}".format(symptom_names[idx], path))
        symptom_model = get_classifier_from_name(args.model, bert_size, hidden_dim=args.hidden_dim)
        try:
            symptom_model.load_state_dict(torch.load(path))
        except Exception as e:
            symptom_model = nn.DataParallel(symptom_model, device_ids=device_list)
            symptom_model.load_state_dict(torch.load(path))

        model_list.append(symptom_model)
    questionnaire_model = QuestionnaireModel(model_list).to("cuda:0")
    questionnaire_model.question_models = nn.ModuleList(model_list)
    questionnaire_model.eval()

    print("Loading data from {}".format(args.input))
    input_dataset = DepressionTextDataset(args.input)
    input_dataloader = DataLoader(input_dataset, batch_size=USER_BATCH_SIZE, shuffle=False, num_workers=4,
                                  collate_fn=collate_fn)
    log_file = open(args.log, "w")
    list_of_users = []
    for user_batch in tqdm(input_dataloader):
        states, scores, labels = questionnaire_model.answer_questionnaire(user_batch, POSTS_PER_USER, tokenizer,
                                                                            bert_model, post_batch_size=POST_BATCH_SIZE)
        assert len(states) == len(scores) == len(labels)
        for idx in range(len(states)):
            st, sc, la = states[idx], scores[idx], labels[idx]
            user = {"states": st, "scores": sc, "label": la}
            list_of_users.append(user)
            for i in range(st.size(0)):
                for j in range(len(symptom_names)):
                    if sc[i, j*2+1] > 0.5:
                        text_without_new_line = user_batch[idx]["posts"][i][1].replace("\t", " ").replace("\n", " ")
                        log_file.write("{}\t{}\t{}\n".format(symptom_names[j], sc[i, j*2+1], text_without_new_line))

    pickle.dump(list_of_users, open(args.output, "wb"))
    log_file.close()
