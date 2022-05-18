import json
import time
import requests
import argparse
import glob
import math
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="retrieve comments given ids")
parser.add_argument("--input_dir", type=str, help="path to the directory of comment ids")
parser.add_argument("--output_dir", type=str, help="the output directory")
parser.add_argument("--type", type=str, help="comment/submission")
args = parser.parse_args()
pushshift_template = "https://api.pushshift.io/reddit/search/{}/?ids={}"

if not os.path.exists(args.input_dir):
    raise "The input directory {} does not exist".format(args.input_dir)

if not args.type in ["comment", "submission"]:
    raise "Type {} is not defined".format(args.type)

files = glob.glob("{}/*.txt".format(args.input_dir))
count_request = 0
failed_files = []
for file in tqdm(files):
    try:
        ids = [line.strip() for line in open(file, "r")]
        n = math.ceil(len(ids) / 100)
        results = []
        for i in range(n):
            param = ",".join(ids[i * 100:(i + 1) * 100])
            pushshift_request = pushshift_template.format(args.type, param)
            response = requests.get(pushshift_request)
            results.extend(response.json()['data'])
            count_request += 1
            if count_request == 100:
                time.sleep(60)
                count_request = 0
        with open(os.path.join(args.output_dir, os.path.basename(file)), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    except Exception as e:
        failed_files.append(file)
json.dump(failed_files, open("{}/failed_files.err".format(args.output_dir), "w"))
