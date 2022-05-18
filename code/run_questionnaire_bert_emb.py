import sys
sys.path.append(sys.path[0] + '/..')
from model.common.dataset import DepressionEmbeddingDataset
import argparse
import torch
import torch.nn as nn
import os
from model.common.utils import get_bert_details_from_alias
from model.common.functional import get_classifier_from_name
from model.questionnaire.questionnaire_model import QuestionnaireModel
from model.common.utils import get_last_hidden_size
import pickle

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

symptom_names = ["anhedonia", "concentration", "eating", "fatigue", "mood", "psychomotor", "self-esteem", "self-harm",
                 "sleep"]
parser = argparse.ArgumentParser("Extract symptom from text")
parser.add_argument("--input", type=str, help="Training file")
parser.add_argument("--bert", type=str, help="bert version")
parser.add_argument("--dim", type=int, default=50, help="dimension of output")
parser.add_argument("--model", type=str, help="type of questionnaire model: linear, cnn, han")
parser.add_argument("--output", type=str, help="Development file")
args = parser.parse_args()

if __name__ == "__main__":
    device_list = list(range(torch.cuda.device_count()))
    print("Loading bert {} model".format(args.bert))
    bert_size, bert_path = get_bert_details_from_alias(args.bert)
    print("Loading symptom model")
    model_list = []
    for idx in range(len(symptom_names)):
        path = os.path.join(*["models", symptom_names[idx], "{}_{}_bert_{}_all.md".format(args.model, args.dim, args.bert)])
        print("Loading {} model from {}".format(symptom_names[idx], path))
        symptom_model = get_classifier_from_name(args.model, bert_size, hidden_dim=args.dim)
        try:
            symptom_model.load_state_dict(torch.load(path))
        except Exception as e:
            symptom_model = nn.DataParallel(symptom_model, device_ids=device_list)
            symptom_model.load_state_dict(torch.load(path))
        model_list.append(symptom_model)
    questionnaire_model = QuestionnaireModel(model_list).to("cuda:0")
    questionnaire_model.eval()

    print("Loading data from {}".format(args.input))
    input_dataset = DepressionEmbeddingDataset(args.input, m_format="pickle", data_field="post")
    new_users = []
    for i in range(len(input_dataset)):
        user, label = input_dataset[i]
        print(user.size(0))
        post_embeddings = user.unsqueeze(1)
        num_posts = post_embeddings.size(0)
        if num_posts < 6:
            post_embeddings = torch.cat([post_embeddings, torch.zeros(6-num_posts, 1,  post_embeddings.size(2))], dim=0)
        scores = []
        states = []
        with torch.no_grad():
            answers = questionnaire_model(post_embeddings)
            for idx in range(len(symptom_names)):
                prob, state = answers[idx]
                prob = prob.cpu()
                state = state.cpu()
                assert prob.size(0) == post_embeddings.size(0) == state.size(0)
                scores.append(prob[:, 1])
                states.append(state)
        scores = torch.stack(scores, dim=1)
        states = torch.cat(states, dim=1)
        assert post_embeddings.size(0) == scores.size(0) == states.size(0)
        assert scores.size(1) == 9
        assert states.size(1) == 9*args["dim"]
        new_user = {"scores": scores, "states": states, "label": label}
        new_users.append(new_user)
    pickle.dump(new_users, open(args.output, "wb"))
