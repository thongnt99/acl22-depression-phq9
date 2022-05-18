import argparse
import os
import json
import numpy as np
import math
parser = argparse.ArgumentParser(description="Parse arguments")
parser.add_argument("--label", type=str)
parser.add_argument("--comment_dir", type=str)
parser.add_argument("--submission_dir", type=str)
args = parser.parse_args()

users = []
label_map = {"0": "control", "1": "depression"}
user_no_post = 0
for line in open(args.label, "r").readlines():
    user_id, label = line.strip().split()
    user_obj = {"id": user_id, "label": label_map[label]}
    posts = []
    p = os.path.join(args.submission_dir, user_id+".txt")
    if os.path.exists(p):
        for post_text in open(p, "r").readlines():
            post_obj = json.loads(post_text)
            posts.append([post_obj["created_utc"], post_obj["selftext"]])
    p = os.path.join(args.comment_dir, user_id+".txt")
    if os.path.exists(p):
        for comment_text in open(p, "r").readlines():
            post_obj = json.loads(comment_text)
            posts.append([post_obj["created_utc"], post_obj["body"]])
    posts = sorted(posts, key=lambda p: p[0])
    if len(posts) == 0:
        print(user_id)
        user_no_post+=1
    else:
        user_obj["posts"] = posts
        users.append(user_obj)
print("Total users {}".format(len(users)))
print("{} users has no post".format(user_no_post))
ids = list(range(len(users)))
ids = np.random.permutation(ids)
num_test = num_dev = math.ceil(len(users)*0.2)
num_train = len(users) - num_test*2
train_users = users[:num_train]
dev_users = users[num_train: -num_dev]
test_users = users[-num_test:]
with open("train.jsonl","w") as f:
    for user in train_users:
        f.write(json.dumps(user)+"\n")
with open("dev.jsonl","w") as f:
    for user in dev_users:
        f.write(json.dumps(user)+"\n")
with open("test.jsonl","w") as f:
    for user in test_users:
        f.write(json.dumps(user)+"\n")
assert len(train_users) + len(dev_users) + len(test_users) == len(users)

