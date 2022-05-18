import json
import argparse
import sys
import re
from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser(description="Code to filter out posts which might contain a symptom")
parser.add_argument("--input", type=str, help="input file")
parser.add_argument("--output", type=str, help="output file")
parser.add_argument("--pp", type=str, help="path to positive patterns file")
parser.add_argument("--np", type=str, help="path to negation template file")
parser.add_argument("--cc", type=str, help="path to conditional clause template")
parser.add_argument("--pt", type=str, help="path to pronoun template")
parser.add_argument("--use_sentiment", action="store_true", default=False, help="whether to use sentiment or not")

# check if the text contains any pattern in the pattern set 
positive_patterns = []


def check_contain_pt(m_sent, pattern_set):
    m_sent = m_sent.lower()
    for pp in pattern_set:
        if pp.search(m_sent):
            return True
    return False


# check if the text's sentiment is positive
# example: "Friends and I stayed up all night playing that amazing game." 
def check_sentiment(m_sent):
    try:
        res_allennlp = allennlp_sentiment_predictor.predict(m_sent)
        res_transformer = transformers_sentiment_predictor(m_sent)[0]
    except:
        return False

    if (res_allennlp['label'] == '1' and res_allennlp['probs'][0] >= 0.8) and (
            res_transformer['label'] == 'POSITIVE' and res_transformer['score'] >= 0.8):
        return True
    else:
        return False
    # check questions


def check_is_question(m_sent):
    m_sent = m_sent.strip().lower()
    if m_sent.endswith(
            "?"):  # or sent.startswith("what ") or sent.startswith("who ") or sent.startswith("whom ") or sent.startswith("where ") or sent.startswith("when ") or sent.startswith("which ") or sent.startswith("whose ") or sent.startswith("how ")
        return True
    return False


# read from text file (line by line)
def read_pattern_file(txt_path):
    res = set()
    with open(txt_path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                res.add(line.strip().lower())
    return res


def get_compiled_positive_pattern(templates):
    return [re.compile(temp) for temp in templates]


def get_compiled_negation_patterns(m_negation_templates, positive_pt_set):
    m_negation_patterns = []
    for template in m_negation_templates:
        for pp in positive_pt_set:
            regex_str = template.replace("{pp}", ".{0,15}" + pp)
            m_negation_patterns.append(re.compile(regex_str))
    return m_negation_patterns


def get_compiled_conditional_clause_patterns(m_cc_templates, positive_pt_set):
    m_conditional_clause_patterns = []
    for cc in m_cc_templates:
        for pp in positive_pt_set:
            m_conditional_clause_patterns.append(re.compile(cc.replace("{pp}", pp)))
    return m_conditional_clause_patterns


def get_compiled_pronoun_patterns(m_pronoun_templates, m_positive_templates):
    m_pronoun_patterns = []
    for pt in m_pronoun_templates:
        for p in m_positive_templates:
            m_pronoun_patterns.append(re.compile(pt.replace("{pp}", p)))
    return m_pronoun_patterns


if __name__ == "__main__":
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    ## read resources
    positive_templates = read_pattern_file(args.pp)
    negation_templates = read_pattern_file(args.np)
    cc_templates = read_pattern_file(args.cc)
    pronoun_templates = read_pattern_file(args.pt)

    positive_patterns = get_compiled_positive_pattern(positive_templates)
    negation_patterns = get_compiled_negation_patterns(negation_templates, positive_templates)
    cc_patterns = get_compiled_conditional_clause_patterns(cc_templates, positive_templates)
    pronoun_patterns = get_compiled_pronoun_patterns(pronoun_templates, positive_templates)
    ## load sentiment models
    if args.use_sentiment:
        ## allennlp sentiment analysis
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.sentiment

        allennlp_sentiment_predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.02.17.tar.gz", cuda_device=0)
        ## transformers from thehuggingface
        from transformers import pipeline

        transformers_sentiment_predictor = pipeline('sentiment-analysis', device=0)

    fout = open(args.output, "w")
    with open(args.input, "r") as f:
        for line in f:
            if not line.strip():
                continue
            post = json.loads(line.strip())
            text = post["text"]
            sents = sent_tokenize(text)
            check_cp = False
            for sent_might_contain_newline in sents:
                for sent in sent_might_contain_newline.split("\n"):
                    tokens = set(sent.lower().split(" "))
                    if ("i" in tokens or "me" in tokens or "my" in tokens) and check_contain_pt(sent,
                                                                                                positive_patterns):
                        if check_is_question(sent):
                            print("question\t{}".format(sent.replace("\n", "@@").replace("\t", " ")))
                            continue
                        if check_contain_pt(sent, negation_patterns):
                            print("negation\t{}".format(sent).replace("\n", "@@").replace("\t", " "))
                            continue
                        if check_contain_pt(sent.lower(), cc_patterns):
                            print("if-clause\t{}".format(sent.replace("\n", "@@").replace("\t", " ")))
                            continue
                        if check_contain_pt(sent.lower(), pronoun_patterns):
                            print("wrong-pronoun\t{}".format(sent.replace("\n", "@@").replace("\t", " ")))
                            continue
                        if args.use_sentiment and check_sentiment(sent):
                            print("positive-sentiment\t{}".format(sent.replace("\n", "@@").replace("\t", " ")))
                            continue
                        check_cp = True
                        target_sent = sent
                        break
                if check_cp:
                    break
            if check_cp:
                post["target_sent"] = target_sent
                fout.write(json.dumps(post))
                fout.write("\n")
    fout.close()
