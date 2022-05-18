from torch import nn
import math
import torch
from model.common.utils import batch_tokenize
from model.common.utils import get_batch_bert_embedding


class QuestionnaireModel(nn.Module):
    def __init__(self, question_models):
        super(QuestionnaireModel, self).__init__()
        self.question_models = nn.ModuleList(question_models)
        self.num_symptoms = len(question_models)

    def forward(self, x_batch):
        res = []
        with torch.no_grad():
            for model in self.question_models:
                y_pred, attention_sate = model(x_batch)
                res.append((y_pred, attention_sate))
        return res

    def answer_questionnaire(self, batch_of_users, posts_per_user, bert_tokenizer,
                             bert_model, post_batch_size=64):
        batch_hidden_states = [] # size = batch_size
        batch_scores = []
        batch_labels = []
        for uid, user in enumerate(batch_of_users):
            posts = [post for _time, post in user["posts"][:posts_per_user]]
            hidden_vector_every_post = []
            score_vector_every_post = []
            for idx in range(math.ceil(len(posts) / post_batch_size)):
                batch_posts = posts[idx * post_batch_size: (idx + 1) * post_batch_size]
                num_posts = len(batch_posts)
                batch_x = batch_tokenize(bert_tokenizer, batch_posts)
                batch_x = get_batch_bert_embedding(bert_model, batch_x)
                # bert_preprocess(tokenizer, bert_model, TOKEN_PER_SENTENCE, batch_posts)
                with torch.no_grad():
                    _outputs = self.forward(batch_x) # [ (batch_size x prob_2, batch_size x hidden) ] x 9
                hidden_vector_every_post.append(torch.cat([h for _, h in _outputs], dim=1))
                score_vector_every_post.append(torch.cat([s for s, _ in _outputs], dim=1))
                # for i in range(self.num_symptoms):
                #     x_batch[uid, idx * post_batch_size: idx * post_batch_size + num_posts, i * last_hidden_dim: (i + 1) * last_hidden_dim] = \
                #         _outputs[i][1].detach().to("cpu")
                #     attention_weight[uid, idx * post_batch_size:idx * post_batch_size + num_posts,
                #     i * last_hidden_dim: (i + 1) * last_hidden_dim] = \
                #         _outputs[i][0][:, 1].unsqueeze(1).expand((-1, last_hidden_dim)).detach().to("cpu")
                #     scores[uid, idx * post_batch_size: (idx+1) * post_batch_size + num_posts] = _outputs[i][0][:, 1]
            user_hidden_matrix = torch.cat(hidden_vector_every_post, dim=0)
            user_score_matrix = torch.cat(score_vector_every_post, dim=0)
            batch_hidden_states.append(user_hidden_matrix)
            batch_scores.append(user_score_matrix)
            if user["label"] == "depression":
                batch_labels.append(1)
            else:
                batch_labels.append(0)

        return batch_hidden_states, batch_scores, batch_labels
