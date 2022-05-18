import argparse 
import random
import json 
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser(description = "Code to debug annotation")
parser.add_argument("--input", type = str, help ="input file")
parser.add_argument("--output", type = str, help = "output file")
parser.add_argument("--k", type = int, help = "number of examples each turns")

args = parser.parse_args()
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

examples = []
with open(args.input, "r") as f:
    for line in f:
        examples.append(line.strip())

samples = random.choices(examples, k = args.k)
fp_count = 0
fout = open(args.output, "w")
for sample in samples:
    post = json.loads(sample)
    if isinstance(post, dict):
        print("=================================")
        print("TEXT: {}".format(post['text']))
        print("=================================")
        print("TARGET: {} {} {}".format(bcolors.OKGREEN, post["target_sent"], bcolors.ENDC))
        print("=================================")
        stt = input("Is this a FALSE-POSIVE case:(y/[n]) ").lower()
        if stt == "y":
            fp_count +=1
            fout.write(post['target_sent']+"\n")
print("{} Estimated FPR is {} {}".format(bcolors.WARNING, fp_count/args.k, bcolors.ENDC ))
fout.close()