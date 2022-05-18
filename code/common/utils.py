import torch
import math


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    bias = bias.transpose(0, 1).unsqueeze(0).expand_as(seq)
    s = torch.bmm(seq, weight.unsqueeze(0).expand(seq.size(0), weight.size(0), weight.size(1))) + bias
    if nonlinearity == 'tanh':
        return torch.tanh(s)
    else:
        return s
    return s


def batch_matmul(seq, weight, nonlinearity=''):
    s = torch.bmm(seq, weight.unsqueeze(0).expand(seq.size(0), weight.size(0), weight.size(1)))
    if nonlinearity == 'tanh':
        return torch.tanh(s)
    else:
        return s


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = rnn_outputs * att_weights
    return torch.sum(attn_vectors, 0).unsqueeze(0)


def batch_tokenize(tokenizer, texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


def get_batch_bert_embedding(bert_model, inputs, trainable=False):
    if not trainable:
        with torch.no_grad():
            output = bert_model(**inputs)[0]
    else:
        output = bert_model(**inputs)[0]
    return output


def split_and_pad_batch(embeddings, tokens_per_sentence):
    batch_size, num_tokens, emb_size = embeddings.size()
    num_sent = math.ceil(num_tokens / tokens_per_sentence)
    inputs = torch.zeros(batch_size, num_sent, tokens_per_sentence, emb_size).to(embeddings.device)
    for i in range(num_sent):
        begin_index = tokens_per_sentence * i
        end_index = min(tokens_per_sentence * (i + 1), num_tokens)
        inputs[:, i, : end_index - begin_index, :] = embeddings[:, begin_index: end_index, :]
    return inputs


def bert_preprocess(bert_tokenizer, bert_model, token_per_sentence, texts):
    inputs = batch_tokenize(bert_tokenizer, texts)
    embeddings = get_batch_bert_embedding(bert_model, inputs)
    reshaped_embeddings = split_and_pad_batch(embeddings, token_per_sentence)
    return reshaped_embeddings


def get_bert_details_from_alias(bert_alias):
    if bert_alias == "tiny":
        bert_path = "google/bert_uncased_L-2_H-128_A-2"
        embedding_size = 128
    elif bert_alias == "mini":
        bert_path = "google/bert_uncased_L-4_H-256_A-4"
        embedding_size = 256
    elif bert_alias == "small":
        bert_path = "google/bert_uncased_L-4_H-512_A-8"
        embedding_size = 512
    elif bert_alias == "medium":
        bert_path = "google/bert_uncased_L-8_H-512_A-8"
        embedding_size = 512
    elif bert_alias == "base":
        bert_path = "google/bert_uncased_L-12_H-768_A-12"
        embedding_size = 768
    elif bert_alias == "large":
        bert_path = "bert-large-uncased"
        embedding_size = 1024
    elif bert_alias == "base-ft":
        bert_path = "/GW/carpet/work/thongnt/reddit_MH/rsdd_ft_bert"
        embedding_size = 768
    elif bert_alias == "base-rdmh":
        bert_path = "/GW/carpet/work/thongnt/reddit_MH/fine-tuned-bert"
        embedding_size = 768
    else:
        raise ValueError("this version does not exists")
    return embedding_size, bert_path


def get_last_hidden_size(name):
    if name == "han":
        return 200
    elif name == "cnn":
        return 250
    elif name == "linear":
        return 50


def f1_loss(y_pred, y_true):
    tp = (y_true * y_pred[:, 1]).float().mean()
    fp = ((1 - y_true) * y_pred[:, 1]).float().mean()
    fn = (y_true * (1 - y_pred[:, 0])).float().mean()
    eps = 1e-10
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    # f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - f1
