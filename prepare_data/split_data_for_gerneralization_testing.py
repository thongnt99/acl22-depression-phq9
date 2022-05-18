import sys
sys.path.append(sys.path[0] + '/../..')
from prepare_data.filter_symptoms import read_pattern_file, get_compiled_positive_pattern, check_contain_pt
import json
import argparse

parser = argparse.ArgumentParser("Split data for gerneralization testing")
parser.add_argument("--inp", type=str, help="question to split")
args = parser.parse_args()

pattern_g1 = "resources/patterns/{}_g1.txt".format(args.question)
pattern_g2 = "resources/patterns/{}_g2.txt".format(args.question)

regex_list_g1 = read_pattern_file(pattern_g1)
pattern_list_g1 = get_compiled_positive_pattern(regex_list_g1)

regex_list_g2 = read_pattern_file(pattern_g2)
pattern_list_g2 = get_compiled_positive_pattern(regex_list_g2)
for part in ["train", "dev", "test"]:
    data_path = "data/{}/{}.jsonl".format(args.question, part)
    g1_f = open("data/{}/{}_g1.jsonl".format(args.question, part), "w")
    g2_f = open("data/{}/{}_g2.jsonl".format(args.question, part), "w")
    with open(data_path, "r") as f:
        for line in f:
            try:
                post = json.loads(line)
            except Exception as e:
                print(e)
            if check_contain_pt(post["text"], pattern_list_g1):
                g1_f.write(line)
            else:
                g2_f.write(line)
    g1_f.close()
    g2_f.close()


