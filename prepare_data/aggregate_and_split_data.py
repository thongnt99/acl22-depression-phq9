import sys

sys.path.append(sys.path[0] + '/..')
import json
import argparse
import random
import numpy as np
from prepare_data.filter_symptoms import get_compiled_positive_pattern, read_pattern_file, check_contain_pt

parser = argparse.ArgumentParser(description="aggregate and split data")
parser.add_argument("--input", type=str, help="input folder")
args = parser.parse_args()
data_dir = "data/{}".format(args.input)
positives = ["mh_filtered_post", "filtered_post"]
negatives = ["neg_posts_2", "neg_posts_3", "negation.neg", "anhedonia.neg", "eating.neg", "mood.neg", "self-esteem.neg",
             "concentration.neg", "fatigue.neg", "psychomotor.neg", "self-harm.neg", "sleep.neg"]
neg_num = np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
neg_posts = []
pos_posts = []
pos_g1_posts = []
pos_g2_posts = []
pattern_g1 = get_compiled_positive_pattern(read_pattern_file("resources/patterns/{}_g1.txt".format(args.input)))
for p in positives:
    with open(data_dir + "/" + p, "r") as f:
        for line in f:
            try:
                post = json.loads(line.strip())
            except:
                continue
            if check_contain_pt(post["text"], pattern_g1):
                pos_g1_posts.append(post["text"])
            else:
                pos_g2_posts.append(post["text"])
            pos_posts.append(post["text"])
neg_num_g1 = neg_num * len(pos_g1_posts)
neg_num_g2 = neg_num * len(pos_g2_posts)
neg_num = neg_num * len(pos_posts)
neg_g1_posts = []
neg_g2_posts = []
for idx, n in enumerate(negatives):
    if n.startswith(args.input):
        continue
    tmp = []
    tmp_g1 = []
    tmp_g2 = []
    print(n)
    with open(data_dir + "/" + n, "r") as f:
        for line in f:
            try:
                post = json.loads(line.strip())
            except:
                continue
            if check_contain_pt(post["text"], pattern_g1):
                tmp_g1.append(post["text"])
            else:
                tmp_g2.append(post["text"])
            tmp.append(post["text"])
            if len(tmp) > neg_num[idx] * 5 and len(tmp_g1) > neg_num_g1[idx]*5 and len(tmp_g2) > neg_num_g2[idx]*5:
                break
    neg_posts.extend(random.choices(tmp, k=int(neg_num[idx])))
    if len(tmp_g1) > 0:
        neg_g1_posts.extend(random.choices(tmp_g1, k=int(neg_num_g1[idx])))
    if len(tmp_g2) > 0:
        neg_g2_posts.extend(random.choices(tmp_g2, k=int(neg_num_g2[idx])))


def permute_ll(l):
    indices = range(len(l))
    indices = np.random.permutation(indices)
    res = [l[idx] for idx in indices]
    return res


pos_posts = permute_ll(pos_posts)
pos_g1_posts = permute_ll(pos_g1_posts)  # np.random.permutation(pos_g1_posts).tolist()
pos_g2_posts = permute_ll(pos_g2_posts)  # np.random.permutation(pos_g2_posts).tolist()
neg_posts = permute_ll(neg_posts)  # np.random.permutation(neg_posts).tolist()
neg_g1_posts = permute_ll(neg_g1_posts)  # np.random.permutation(neg_g1_posts).tolist()
neg_g2_posts = permute_ll(neg_g2_posts)  # np.random.permutation(neg_g2_posts).tolist()

train = []
train_g1 = []
train_g2 = []
dev = []
dev_g1 = []
dev_g2 = []
test = []
test_g1 = []
test_g2 = []


def get_part(X, y, part):
    if part == "train":
        l = 0
        r = round(len(X) * 0.7)
    elif part == "dev":
        l = round(len(X) * 0.7)
        r = round(len(X) * 0.85)
    elif part == "test":
        l = round(len(X) * 0.85)
        r = len(X)
    res = [{"text": post, "label": y} for post in X[l:r]]
    return res


train.extend(get_part(pos_posts, 1, "train"))
train.extend(get_part(neg_posts, 0, "train"))
dev.extend(get_part(pos_posts, 1, "dev"))
dev.extend(get_part(neg_posts, 0, "dev"))
test.extend(get_part(pos_posts, 1, "test"))
test.extend(get_part(neg_posts, 0, "test"))

train_g1.extend(get_part(pos_g1_posts, 1, "train"))
train_g1.extend(get_part(neg_g1_posts, 0, "train"))
dev_g1.extend(get_part(pos_g1_posts, 1, "dev"))
dev_g1.extend(get_part(neg_g1_posts, 0, "dev"))
test_g1.extend(get_part(pos_g1_posts, 1, "test"))
test_g1.extend(get_part(neg_g1_posts, 0, "test"))

train_g2.extend(get_part(pos_g2_posts, 1, "train"))
train_g2.extend(get_part(neg_g2_posts, 0, "train"))
dev_g2.extend(get_part(pos_g2_posts, 1, "dev"))
dev_g2.extend(get_part(neg_g2_posts, 0, "dev"))
test_g2.extend(get_part(pos_g2_posts, 1, "test"))
test_g2.extend(get_part(neg_g2_posts, 0, "test"))


def write(data, file_name):
    with open(file_name, "w") as f:
        for post in data:
            f.write(json.dumps(post) + "\n")


write(train, data_dir + "/train.jsonl")
write(dev, data_dir + "/dev.jsonl")
write(test, data_dir + "/test.jsonl")
write(train_g1, data_dir + "/train_g1.jsonl")
write(dev_g1, data_dir + "/dev_g1.jsonl")
write(test_g1, data_dir + "/test_g1.jsonl")
write(train_g2, data_dir + "/train_g2.jsonl")
write(dev_g2, data_dir + "/dev_g2.jsonl")
write(test_g2, data_dir + "/test_g2.jsonl")

print("Done!")
print("Train\tDev\tTest")
print("{}\t{}\t{}".format(len(train), len(dev), len(test)))

print("Train G1\tDev G1\tTest G1")
print("{}\t{}\t{}".format(len(train_g1), len(dev_g1), len(test_g1)))

print("Train G2\tDev G2\tTest G2")
print("{}\t{}\t{}".format(len(train_g2), len(dev_g2), len(test_g2)))
