import sys
sys.path.append(sys.path[0] + '/..')
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.common.depression_dataset import RSDD
from model.common.depression_dataset import collate_fn
from model.questionnaire.pb_questionnaire_model import PatternBasedQuestionnaireModel
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

symptom_names = ["anhedonia", "concentration", "eating", "fatigue", "mood", "psychomotor", "self-esteem", "self-harm",
                 "sleep"]
parser = argparse.ArgumentParser("Extract symptom from text")
parser.add_argument("--input", type=str, help="INP file")
parser.add_argument("--output", type=str, help="OUT file")
args = parser.parse_args()

if __name__ == "__main__":
    questionnaire_model = PatternBasedQuestionnaireModel()
    print("Loading data from {}".format(args.input))
    input_dataset = RSDD(args.input)
    input_dataloader = DataLoader(input_dataset, batch_size=USER_BATCH_SIZE, shuffle=False, num_workers=4,
                                  collate_fn=collate_fn)
    last_hidden_dim = get_last_hidden_size(args.model)
    output_file = open(args.output, "w")
    count_examples = 0
    log_file = open("logs.txt")
    for user_batch in tqdm(input_dataloader):
        x_batch, y_batch, scores = questionnaire_model.answer_questionnaire(user_batch)
        for idx in range(y_batch.size(0)):
            count_examples += 1
            x = x_batch[idx]
            y = y_batch[idx]
            score = scores[idx]
            post = {"input": x.tolist(), "scores": score.tolist(), "label": y.item()}
            for i in range(400):
                for j in range(len(symptom_names)):
                    if score[i][j] > 0.5:
                        log_file.write("++++++++++++++++++++{}++++{}++++++++++++++++++\n".format(symptom_names[j], score[i][j]))
                        log_file.write(user_batch[idx]["posts"][i][1])
                        log_file.write("\n\n")
                        log_file.flush()
            output_file.write("{}\n".format(json.dumps(post)))
    assert len(input_dataset) == count_examples
    output_file.close()
