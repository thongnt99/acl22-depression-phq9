from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from model.questionnaire.han.han_classifier import HANClassifier
from model.questionnaire.cnn.cnn_classifier import CNNClassifier
from model.questionnaire.hybrid.hybrid_classifier import HybridClassifier
from model.questionnaire.linear.linear_classifier import LinearClassifier
from model.questionnaire.linear.linear_classifier_attention import BertLinearClassifier


def evaluate(model, data_loader, criterion, device="cuda:0", threshold=0.5, to_print=True, return_results=False):
    model.eval()
    with torch.no_grad():
        eval_true = []
        eval_pred = []
        eval_loss = []
        eval_score = []
        for batch in tqdm(data_loader):
            batch_x, batch_y_true = batch
            batch_y_true = batch_y_true.to(device)
            preds, _ = model(batch_x)
            if preds.size(1) == 2:
                batch_y_pred = preds[:,1]>threshold
                loss = criterion(preds, batch_y_true)
            else:
                batch_y_pred = preds > threshold
                loss = criterion(preds.squeeze(), batch_y_true.float().squeeze())
            if not loss.isnan():
                eval_loss.append(loss.detach().item())
            else:
                print(loss)
            eval_true = eval_true + batch_y_true.tolist()
            eval_pred = eval_pred + batch_y_pred.tolist()
            eval_score = eval_score + preds[:,1].tolist()
    if to_print:
        print(classification_report(eval_true, eval_pred))
    eval_f1 = f1_score(eval_true, eval_pred)
    eval_true = torch.tensor(eval_true)
    eval_pred = torch.tensor(eval_pred)
    mean_acc = (eval_true == eval_pred).float().mean()
    if to_print:
        print(f'roca auc {roc_auc_score(eval_true, eval_score)}')
    with open("prediction.txt","w") as f:
        for t, p in zip(eval_true, eval_score):
            f.write("{}\t{}\n".format(t,p))
    model.train(True)
    if not return_results:
        return mean_acc, torch.tensor(eval_loss).mean(), eval_f1
    else:
        return mean_acc, torch.tensor(eval_loss).mean(), eval_f1, eval_score, eval_true

def train(model, train_dl, dev_dl, optimizer, criterion, lr_scheduler, num_epoch, es_patience, model_path,
          device="cuda:0", threshold=0.5, one_cycle_lr=False, regularize=False, regularize_weight=0):
    # smallest_loss = 100
    highest_f1 = -1
    decline_count = 0
    for epoch_idx in range(num_epoch):
        batch_loss = []
        batch_acc = []
        for batch_idx, batch in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            batch_x, batch_y = batch
            batch_y = batch_y.to(device)
            preds, _ = model(batch_x)
            if preds.size(1) == 2:
                pred_y = preds.argmax(1)
                loss = criterion(preds, batch_y)
            else:
                pred_y = preds > threshold
                loss = criterion(preds.squeeze(), batch_y.float().squeeze())
            acc = (pred_y == batch_y).float().mean()
            if regularize:  
                l2 = torch.tensor(0.0).to("cuda:0")
                l2.requires_grad = True
                for conv in model.module.convs:
                    l2 = l2 + conv.weight.norm(2)
                loss =  loss + regularize_weight*l2

            loss.backward()
            if (loss.isnan()):
                print(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            batch_loss.append(loss.detach().item())
            batch_acc.append(acc)
            if one_cycle_lr:
                lr_scheduler.step()
        if not one_cycle_lr:
            lr_scheduler.step()
        train_acc, train_loss = torch.tensor(batch_acc).mean(), torch.tensor(batch_loss).mean()
        dev_acc, dev_loss, dev_f1 = evaluate(model, dev_dl, criterion, device=device, threshold=threshold)
        print("Epoch {} train_loss {} train_acc {} dev_acc {}, dev_loss {}, dev_f1 {}".format(epoch_idx, train_loss,
                                                                                              train_acc, dev_acc,
                                                                                              dev_loss, dev_f1))
        # if dev_loss < smallest_loss:
        if dev_f1 > highest_f1:
            # path = os.path.join(saving_path, "han_tiny_bert.md")
            print("Saving model to {}".format(model_path))
            torch.save(model.state_dict(), model_path)
            # smallest_loss = dev_loss
            highest_f1 = dev_f1
            decline_count = 0
        else:
            decline_count += 1
            if decline_count == es_patience:
                print("Early stopping at epoch {}".format(epoch_idx))
                return highest_f1
    return highest_f1


def test(model, dataloader, criterion, device="cuda:0", threshold=0.5, to_print=False, return_results=False):
    if to_print:
        print("Evaluating on test set with threshold {} ..".format(threshold))
    if not return_results:
        test_acc, test_loss, f1 = evaluate(model, dataloader, criterion, device=device, threshold=threshold, to_print=to_print, return_results=return_results)
        return test_acc, test_loss, f1
    else:
        test_acc, test_loss, f1, test_score, test_true = evaluate(model, dataloader, criterion, device=device, threshold=threshold, to_print=to_print, return_results=return_results)
        return test_acc, test_loss, f1, test_score, test_true


def get_classifier_from_name(bert_path, name, bert_dim, hidden_dim=50):
    if name == "han":
        return HANClassifier(bert_dim)
    elif name == "cnn":
        if hidden_dim == 20:
            return CNNClassifier(bert_dim, 10, [3, 4], 2, 0.1)
        elif hidden_dim == 5:
            return CNNClassifier(bert_dim, 1, [2, 3, 4, 5, 6], 2, 0.1)
        else:
            return CNNClassifier(bert_dim, 50, [3, 4, 5, 6, 7], 2, 0.1)
    elif name == "linear":
        return LinearClassifier(bert_dim, hidden_dim)
    elif name == "linear_bert":
        return BertLinearClassifier(bert_path, bert_dim, hidden_dim)
    elif name == "hybrid":
        return HybridClassifier(bert_dim, 1, [2, 3, 4, 5, 6], 2, 0.1)
    else:
        raise ValueError("model was not defined. User: han, cnn, linear")
