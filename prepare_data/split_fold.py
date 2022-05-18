import argparse
import json
import numpy as np
parser = argparse.ArgumentParser("Split training data in to folds")
parser.add_argument("--input", type=str, help="Input file to split.")
parser.add_argument("--n_fold", type=int, help="Number of folds to split")
args = parser.parse_args()
all_posts = []
with open("data/{}/train.jsonl".format(args.input), "r") as f:
    for line in f:
        post = json.loads(line)
        all_posts.append(post)

indices = list(range(len(all_posts)))
indices = np.random.permutation(indices)
fold_size = len(all_posts)//args.n_fold
rights = [0] + [fold_size*(i+1) for i in range(args.n_fold)]
rights[-1] = len(all_posts)
for i in range(args.n_fold):
    out_file = "data/{}/train_fold_{}".format(args.input, i+1)
    with open(out_file, "w") as f:
        for idx in indices[rights[i]: rights[i+1]]:
            f.write(json.dumps(all_posts[idx]))
