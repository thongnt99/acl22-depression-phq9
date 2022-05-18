import torch
import os
from nltk.tokenize import sent_tokenize
from prepare_data.filter_symptoms import read_pattern_file,check_is_question, check_contain_pt, \
    get_compiled_pronoun_patterns, get_compiled_conditional_clause_patterns, get_compiled_positive_pattern, \
    get_compiled_negation_patterns

# resource dir
resource_dir = "resources/patterns/"
# positive pattern
pp_list = ["anhedonia_problem.txt", "concentration_problem.txt", "eating_problem.txt",
           "fatigue_problem.txt", "mood_problem.txt", "psychomotor_problem.txt",
           "self-esteem_problem.txt", "self-harm_problem.txt", "sleep_disorder.txt"]
# negation template
np_path = "negation_templates.txt"
# conditional clause
cc_path = "cc_templates.txt"
# pronoun templates
pt_path = "pronoun_templates.txt"


def get_path(resource_name):
    return os.path.join(resource_dir, resource_name)


class PatternBasedQuestionnaireModel:
    def __init__(self):
        super(PatternBasedQuestionnaireModel, self).__init__()
        list_of_positive_templates = [read_pattern_file(get_path(path)) for path in pp_list]
        negation_templates = read_pattern_file(get_path(np_path))
        cc_templates = read_pattern_file(get_path(cc_path))
        pronoun_templates = read_pattern_file(get_path(pt_path))
        self.list_pp = [get_compiled_positive_pattern(p) for p in list_of_positive_templates]
        self.list_np = [get_compiled_negation_patterns(negation_templates, p) for p in list_of_positive_templates]
        self.list_cc = [get_compiled_conditional_clause_patterns(cc_templates, p) for p in list_of_positive_templates]
        self.list_prp = [get_compiled_pronoun_patterns(pronoun_templates, p) for p in list_of_positive_templates]

    def is_match(self, post, idx):
        sents = sent_tokenize(post)
        check_cp = False
        for sent_might_contain_newline in sents:
            for sent in sent_might_contain_newline.split("\n"):
                tokens = set(sent.lower().split(" "))
                if ("i" in tokens or "me" in tokens or "my" in tokens) and check_contain_pt(sent, self.list_pp[idx]):
                    if check_is_question(sent):
                        continue
                    if check_contain_pt(sent, self.list_np[idx]):
                        continue
                    if check_contain_pt(sent.lower(), self.list_cc[idx]):
                        continue
                    if check_contain_pt(sent.lower(), self.list_prp[idx]):
                        continue
                    check_cp = True
                    break
            if check_cp:
                break
        return 1 if check_cp is True else 0

    def answer_questionnaire(self, batch_of_users, post_per_user=1500):
        scores = torch.zeros(len(batch_of_users), post_per_user, 9).to("cpu")
        y_batch = []
        for u_id, user in enumerate(batch_of_users):
            posts = [post[1] for post in user["posts"][:post_per_user]]
            for p_id, post in enumerate(posts):
                for t_id in range(len(self.list_cc)):
                    scores[u_id, p_id, t_id] = self.is_match(post, t_id)
            if user["label"] == "depression":
                y_batch.append(1)
            else:
                y_batch.append(0)
        return torch.tensor([]), torch.tensor(y_batch), scores
