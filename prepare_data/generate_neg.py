import argparse 
import json 
import re
import sys
import numpy as np
from filter_symptoms import read_pattern_file
from filter_symptoms import check_contain_positive_pt
from filter_symptoms import check_contain_negative_pt

parser = argparse.ArgumentParser(description="Code to generate negative training examples")
parser.add_argument("--input", type = str, help ="path to input file")
parser.add_argument("--output", type = str, help ="path to output file")
subparsers = parser.add_subparsers(help = 'selecting method')

parser_1 = subparsers.add_parser("1", help ="collect post contain keywords but not positive patterns")
parser_1.add_argument("--kw", type = str, help ="path to the symptom's keywords")
parser_1.add_argument("--pp", type = str, help ="path to the positive pattern")
parser_1.add_argument("--np", type = str, help ="path to the negative patterns")
parser_1.set_defaults(meth=1)

parser_2 = subparsers.add_parser("2", help ="change pronouns")
parser_2.set_defaults(meth=2)

parser_3 = subparsers.add_parser("3", help ="data from other symptoms. For example, posts from eating disorder which doesn't contain sleep")
parser_3.set_defaults(meth=3)
parser_3.add_argument("--kw", type =str, help ="path to symptom's keywords")

parser_4 = subparsers.add_parser("4", help = "negation positive patterns")
parser_4.set_defaults(meth=4)
parser_4.add_argument("--neg_map", type = str, help = "negation mapping")


if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(0)

keyword_patterns = []
def check_contain_keywords(sent, keyword_set, strict_matching = False):
    if (len(keyword_patterns) == 0):
        for keyword in keyword_set:
            if strict_matching:
                keyword_patterns.append(re.compile("(?<![a-zA-Z])"+keyword+"(?![a-zA-Z])")) #add the word boundary at the beginning and at the end 
            else:
                keyword_patterns.append(re.compile(keyword))
    for pt in keyword_patterns:
        if pt.search(sent):
            return True 
    return False 

args = parser.parse_args()
if (args.meth == 1):
    keywords = read_pattern_file(args.kw)
    positive_patterns = read_pattern_file(args.pp)
    negative_patterns = read_pattern_file(args.np)
    fout = open(args.output,"w")
    with open(args.input, "r") as f:
        for line in f:
            if (line.strip() == ""):
                continue
            post = json.loads(line)
            text = post["text"]
            if check_contain_keywords(text, keywords):
                if not check_contain_positive_pt(text,positive_patterns) or check_contain_negative_pt(text, positive_patterns, negative_patterns):
                    fout.write(json.dumps(post)+"\n")

    fout.close() 
elif (args.meth ==2):
    # replace pronouns
    #first person pronoun
    fp_pronoun= [re.compile("(?<![a-zA-Z])(i|I)(?![a-zA-Z])"), re.compile("(?<![a-zA-Z])my(?![a-zA-Z])"), re.compile("(?<![a-zA-Z])myself(?![a-zA-Z])"), re.compile("(?<![a-zA-Z])me(?![a-zA-Z])")]
    pronoun_he = ["he","his","himself","him"]
    pronoun_she = ["she","her","herself","her"]
    pronoun_they = ["they","their","themselves","them"]
    p1 = re.compile("(?<![a-zA-Z])he(?![a-zA-Z])")
    p2 = re.compile("(?<![a-zA-Z])she(?![a-zA-Z])")
    male_names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth", "Joshua", "George", "Kevin", "Brian", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Stephen", "Jonathan", "Larry", "Justin", "Scott", "Brandon", "Frank", "Benjamin", "Gregory", "Samuel", "Raymond", "Patrick", "Alexander", "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Henry", "Douglas", "Adam", "Peter", "Nathan", "Zachary", "Walter", "Kyle", "Harold", "Carl", "Jeremy", "Keith", "Roger", "Gerald", "Ethan", "Arthur", "Terry", "Christian", "Sean", "Lawrence", "Austin", "Joe", "Noah", "Jesse", "Albert", "Bryan", "Billy", "Bruce", "Willie", "Jordan", "Dylan", "Alan", "Ralph", "Gabriel", "Roy", "Juan", "Wayne", "Eugene", "Logan", "Randy", "Louis", "Russell", "Vincent", "Philip", "Bobby", "Johnny", "Bradley","my brother", "my son", "my dad", "my cousin","my father", "the man"]
    female_names = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Margaret", "Lisa", "Betty", "Dorothy", "Sandra", "Ashley", "Kimberly", "Donna", "Emily", "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Laura", "Sharon", "Cynthia", "Kathleen", "Helen", "Amy", "Shirley", "Angela", "Anna", "Brenda", "Pamela", "Nicole", "Ruth", "Katherine", "Samantha", "Christine", "Emma", "Catherine", "Debra", "Virginia", "Rachel", "Carolyn", "Janet", "Maria", "Heather", "Diane", "Julie", "Joyce", "Victoria", "Kelly", "Christina", "Joan", "Evelyn", "Lauren", "Judith", "Olivia", "Frances", "Martha", "Cheryl", "Megan", "Andrea", "Hannah", "Jacqueline", "Ann", "Jean", "Alice", "Kathryn", "Gloria", "Teresa", "Doris", "Sara", "Janice", "Julia", "Marie", "Madison", "Grace", "Judy", "Theresa", "Beverly", "Denise", "Marilyn", "Amber", "Danielle", "Abigail", "Brittany", "Rose", "Diana", "Natalie", "Sophia", "Alexis", "Lori", "Kayla", "Jane", "my sister", "my aunt","my daughter", "my mom","the woman"]
    fout = open(args.output,"w")
    with open(args.input,"r") as f:
        for line in f:
            post = json.loads(line.strip())
            if isinstance(post,dict):
                text = post["text"]
                ## I -> He
                he_text = text
                for idx, p in enumerate(fp_pronoun):
                    he_text = p.sub(pronoun_he[idx],he_text)
                post["text"] = he_text
                fout.write(json.dumps(post)+"\n")
                ## I -> She
                she_text = text 
                for idx, p in enumerate(fp_pronoun):
                    she_text = p.sub(pronoun_she[idx], she_text)
                post["text"] = he_text
                fout.write(json.dumps(post)+"\n")
                ## I -> They 
                they_text = text 
                for idx, p in enumerate(fp_pronoun):
                    they_text = p.sub(pronoun_they[idx], they_text)
                post["text"] = they_text
                fout.write(json.dumps(post)+"\n")
                ## I -> Male Name
                if (p1.search(he_text)):
                    male_text = p1.sub(np.random.choice(male_names,1)[0], he_text)
                    post["text"] = male_text
                    fout.write(json.dumps(post)+'\n')
                if (p2.search(she_text)):
                    female_text = p1.sub(np.random.choice(female_names,1)[0], she_text)
                    post["text"] = female_text
                    fout.write(json.dumps(post)+'\n')
    fout.close()
elif (args.meth ==3):
    fout = open(args.output,"w")
    keywords = read_pattern_file(args.kw)
    with open(args.input, "r") as f:
        for line in f:
            post = json.loads(line.strip())
            if isinstance(post,dict) and "text" in post:
                if not check_contain_keywords(post["text"], keywords, strict_matching = False):
                    fout.write(json.dumps(post)+'\n')
    fout.close()
elif (args.meth == 4):
    from nltk.tokenize import sent_tokenize 
    fout = open(args.output,"w")
    neg_map = {}
    with open(args.neg_map,"r") as f:
        for line in f:
            pos, neg = line.strip().split("\t")
            neg_map[pos] = neg
    with open(args.input, "r") as f:
        for line in f:
            post = json.loads(line.strip())
            if isinstance(post,dict) and "text" in post:
                text = post["text"]
                sents = sent_tokenize(text)
                for sent_might_contain_newline in sents:
                    for sent in sent_might_contain_newline.split("\n"):
                        flag = False
                        for key in neg_map:
                            p = re.compile(key)
                            if p.search(sent):
                                sent = re.sub(p,neg_map[key],sent)
                                flag = True 
                        if (flag):
                            new_post = {"text": sent}
                            fout.write(json.dumps(new_post)+"\n")
                            break
    fout.close()    
