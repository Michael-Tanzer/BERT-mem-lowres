import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import wandb

from pytorch_transformers import BertModel, AdamW, WarmupLinearSchedule
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import trange

from .model_utils import convert_examples_to_features
from .consts import get_data_dirs_cardinal, get_processors, get_tokenizers, get_configs, get_reporters


use_wandb = True
proto = True
bert_type = "True"
lr = 1e-4
negative_sample_ratio = 2
warmup_proportion = 0.1
total_steps = 756
dataset = "jnlpba-25dna"
embedding_dimension = 300
reduced_labels = (2, 3)
support_min = 10
support_other = 100

num_labels = 14


def setup_optimizer_and_scheduler(model, bert, lr, warmup_proportion=0.1, total_steps=756):
    param_optimizer = []

    param_optimizer += list(model.named_parameters())
    param_optimizer += list(bert.bert.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params':       [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(warmup_proportion * total_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    return optimizer, scheduler


def get_labels(logits, label_ids, label_map):
    y_true = []
    y_pred = []
    logits = torch.argmax(logits, dim=2)

    for i, label in enumerate(label_ids):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(label):
            if j == 0:
                continue
            elif label_ids[i][j] == len(label_map) - 1:
                y_true.append(temp_1)
                y_pred.append(temp_2)
                break
            else:
                temp_1.append(label_map[label_ids[i][j].item()])
                temp_2.append(label_map[logits[i][j].item()])

    return y_true, y_pred


def get_token_report(metrics):
    headers = ["class", "precision", "recall", "f1score", "support"]
    classes = get_processors()[dataset]().get_labels()
    classes.remove("O"); classes.remove("[SEP]"); classes.remove("[CLS]")

    rows = []
    for i, c in enumerate(classes):
        mets = [metrics[j][i] for j in range(len(metrics))]
        rows.append([c] + mets)

    return headers, rows


def log_all(y_pred, y_true, prefix, log_token, log_entity, step):
    if log_token and log_entity:
        raise ValueError("Not supported at the moment :(")

    if log_token:
        pred = y_pred.argmax(dim=1)
        prf_micro = precision_recall_fscore_support(y_true.cpu(), pred.cpu(), labels=range(1, num_labels - 3), average='micro', zero_division=0)
        prf_weighted = precision_recall_fscore_support(y_true.cpu(), pred.cpu(), labels=range(1, num_labels - 3), average='weighted', zero_division=0)
        prf_macro = precision_recall_fscore_support(y_true.cpu(), pred.cpu(), labels=range(1, num_labels - 3), average='macro', zero_division=0)
        auc_macro = roc_auc_score(y_true, torch.softmax(y_pred, dim=1), multi_class='ovo', labels=range(num_labels - 3), average="macro")
        auc_weighted = roc_auc_score(y_true, torch.softmax(y_pred, dim=1), multi_class='ovo', labels=range(num_labels - 3), average="weighted")
        if use_wandb:
            wandb.log({
                f"token/{prefix}-micro-precision": prf_micro[0],
                f"token/{prefix}-micro-recall":    prf_micro[1],
                f"token/{prefix}-micro-f1score":   prf_micro[2],
                f"token/{prefix}-weighted-precision": prf_weighted[0],
                f"token/{prefix}-weighted-recall":    prf_weighted[1],
                f"token/{prefix}-weighted-f1score":   prf_weighted[2],
                f"token/{prefix}-macro-precision": prf_macro[0],
                f"token/{prefix}-macro-recall":    prf_macro[1],
                f"token/{prefix}-macro-f1score":   prf_macro[2],
                f"token/{prefix}-macro-auc":   auc_macro,
                f"token/{prefix}-weighted-auc":   auc_weighted,
        }, step=step)

        if prefix == "test" and use_wandb:
            headers, data = get_token_report(precision_recall_fscore_support(y_true.cpu(), pred.cpu(), labels=range(1, num_labels - 3), average=None, zero_division=0))
            wandb.log({f"token/{prefix}-report": wandb.Table(columns=headers, data=data)}, step=step)

        return prf_micro[2]

    if log_entity:
        print("logging entities")
        if dataset == "wnut17":
            task = "wnut"
        else:
            task = "ner"
        metrics_micro = get_reporters()[task](y_true, y_pred, average='micro')
        metrics_weighted = get_reporters()[task](y_true, y_pred, average='weighted')
        metrics_macro = get_reporters()[task](y_true, y_pred, average='macro')
        if use_wandb: wandb.log({
                f"entity/{prefix}-micro-precision": metrics_micro["precision"],
                f"entity/{prefix}-micro-recall":    metrics_micro["recall"],
                f"entity/{prefix}-micro-f1score":   metrics_micro["f1score"],
                f"entity/{prefix}-macro-precision": metrics_macro["precision"],
                f"entity/{prefix}-macro-recall":    metrics_macro["recall"],
                f"entity/{prefix}-macro-f1score":   metrics_macro["f1score"],
                f"entity/{prefix}-weighted-precision": metrics_weighted["precision"],
                f"entity/{prefix}-weighted-recall":    metrics_weighted["recall"],
                f"entity/{prefix}-weighted-f1score":   metrics_weighted["f1score"],
                f"entity/{prefix}-report": wandb.Table(columns=[""] + metrics_micro["report"].split("\n")[0].split(),
                                                             data=[t.split() for i, t in enumerate(metrics_micro["report"].split("\n")) if
                                                                   (0 < i < len(metrics_micro["report"].split("\n")) - 3) and (len(t) > 0)])
        }, step=step)
        return metrics_micro["f1score"]


def get_data(examples, label_list, label_map, tokenizer, label_noise_addition=0.0):
    features = convert_examples_to_features(examples, label_list, 50, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    noise_selector = torch.rand_like(all_label_ids, dtype=torch.float) < label_noise_addition

    if 0 < label_noise_addition <= 1:
        noise = torch.zeros_like(all_label_ids, dtype=torch.float)
        noise.uniform_(to=len(label_list) + 1)
        noise = noise.long()
        inverse_map = {v: k for k, v in label_map.items()}
        noise_selector[all_input_ids == inverse_map['[CLS]']] = False
        noise_selector[all_input_ids == inverse_map['[PAD]']] = False
        noise_selector[all_input_ids == inverse_map['[SEP]']] = False
        noise_selector[noise == all_label_ids] = False
        all_label_ids[noise_selector] = noise[noise_selector]

        torch.save(noise_selector.long().view(-1), 'noise_mask.pt')

    idxs = torch.tensor([int(i) for i in range(all_input_ids.view(-1).shape[0])]).view_as(all_input_ids)

    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    data = {
            "input_ids":      all_input_ids,
            "input_mask":     all_input_mask,
            "segment_ids":    all_segment_ids,
            "label_ids":      all_label_ids,
            "valid_ids":      all_valid_ids,
            "lmask_ids":      all_lmask_ids,
            "noise_selector": noise_selector.long(),
            "absolute_ids":   idxs.long()
    }

    return data


class MockTrainedBert:
    def __init__(self, dataset):
        self.dataset = dataset
        self.nth_layer = 12
        self.train_embeddings = torch.load(f"./train_embeddings_{self.dataset}_2_{self.nth_layer}.pt").view(-1, 50, 768)  # pre-computed from BERT training
        self.train_logits = torch.load(f"./train_logits_{self.dataset}_2_{self.nth_layer}.pt").view(-1, num_labels, 50)  # pre-computed from BERT training
        self.test_embeddings = torch.load(f"./test_embeddings_{self.dataset}_2_{self.nth_layer}.pt").view(-1, 50, 768)  # pre-computed from BERT training
        self.test_logits = torch.load(f"./test_logits_{self.dataset}_2_{self.nth_layer}.pt").view(-1, num_labels, 50)  # pre-computed from BERT training
        self.train_labels = torch.load(f"./train_labels_{self.dataset}_2_{self.nth_layer}.pt").view(-1, 50)  # pre-computed from BERT training
        self.test_labels = torch.load(f"./test_labels_{self.dataset}_2_{self.nth_layer}.pt").view(-1, 50)  # pre-computed from BERT training

    def get_from_sent_idxs(self, idxs, train=True, output_valid=False):
        if output_valid and train:
            return self.train_embeddings[idxs], self.train_labels[idxs], self.train_embeddings[idxs], self.train_labels[idxs]
        elif output_valid and not train:
            return self.test_embeddings[idxs], self.test_labels[idxs], self.test_embeddings[idxs], self.test_labels[idxs]
        elif not output_valid and train:
            return self.train_embeddings[idxs], self.train_labels[idxs]
        else:
            return self.test_embeddings[idxs], self.test_labels[idxs]


class TrueBert:
    def __init__(self, dataset):
        self.data_dir = get_data_dirs_cardinal()[dataset]
        self.processor = get_processors()[dataset]()
        self.label_list = self.processor.get_labels()
        self.label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        self.label_map[0] = '[PAD]'
        self.tokenizer = get_tokenizers()['BERT'].from_pretrained('bert-base-cased', do_lower_case=False)
        self.train_examples = self.processor.get_train_examples(self.data_dir)
        self.test_examples = self.processor.get_test_examples(self.data_dir)
        self.train_data = get_data(self.train_examples, self.label_list, self.label_map, self.tokenizer)
        self.test_data = get_data(self.test_examples, self.label_list, self.label_map, self.tokenizer)

        config = get_configs()["BERT"].from_pretrained('bert-base-cased', num_labels=len(self.label_map), finetuning_task='ner',
                                                       output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-cased', from_tf=False, config=config).cuda()

    def get_from_sent_idxs(self, idxs, train=True, output_valid=False):
        from copy import deepcopy
        if train:
            data = {k: deepcopy(v[idxs]) for k, v in self.train_data.items()}
        else:
            data = {k: v[idxs] for k, v in self.test_data.items()}

        bert_output = self.bert(data["input_ids"].cuda(), data["segment_ids"].cuda(), data["input_mask"].cuda(), head_mask=None)
        sequence_output = bert_output[0]

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if data["valid_ids"][i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        active_loss = data["lmask_ids"].view(-1) == 1
        active_output = valid_output.view(-1, 768)[active_loss]
        active_labels = data["label_ids"].view(-1)[active_loss]

        if output_valid:
            return active_output, active_labels, valid_output, data['label_ids']
        else:
            return active_output, active_labels


class ProtoNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, bo=4):
        super().__init__()

        self.embedding_dim = embedding_dimension

        self.model = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(input_dim, embedding_dim)
        )

        self.bo = nn.Parameter(-torch.ones(1) * bo, requires_grad=True)

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.model(x)


class StandardNetwork(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super().__init__()

        self.model = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(embedding_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train_proto(y, test_data, bert_model, epochs=1000, lr=1e-3, negative_sampling_ratio=4, l1=2, l2=3, support_min=10, support_other=100,
                embedding_dimension=300):
    loss_f = nn.CrossEntropyLoss()

    s1 = support_min
    s2 = support_other
    alpha = 0.1
    running_mean = False

    idxs_sent_min = set()
    idxs_sent_other = set()

    for i, labs in enumerate(y):
        found = False
        for l in labs:
            if l == l1 or l == l2:
                found = True
        if found:
            idxs_sent_min.add(i)
        else:
            idxs_sent_other.add(i)

    if len(idxs_sent_min)/2 < s1 and len(idxs_sent_min) > 0:
        s1 = len(idxs_sent_min)//2

    model = ProtoNetwork(768, embedding_dimension, bo=4).cuda()
    print(torch.cuda.memory_allocated())
    optimizer, scheduler = setup_optimizer_and_scheduler(model, bert_model, lr, warmup_proportion=warmup_proportion, total_steps=total_steps)

    idxs_sent_min = list(idxs_sent_min)
    idxs_sent_other = list(idxs_sent_other)

    current_val_max = 0
    current_val_max_times = 0

    if running_mean:
        all_centers = [None] * (num_labels - 4)

    for epoch in trange(epochs):
        model.train()
        bert.bert.train()

        if len(idxs_sent_min) == 0:
            support_min = torch.zeros((0, embedding_dimension)).cuda()
            query_min = torch.zeros((0, embedding_dimension)).cuda()
            support_min_labels = torch.zeros(0, dtype=torch.long)
            query_min_labels = torch.zeros(0, dtype=torch.long)
            min_idxs = None
            support_min_bert = None
            query_min_bert = None
        else:
            min_idxs = torch.LongTensor(idxs_sent_min)[torch.randperm(len(idxs_sent_min))]
            support_min_bert, support_min_labels = bert_model.get_from_sent_idxs(min_idxs[:s1])
            support_min = model(support_min_bert.view(-1, 768))
            query_min_bert, query_min_labels = bert_model.get_from_sent_idxs(min_idxs[s1:min(len(min_idxs) - 1, 25)])
            query_min = model(query_min_bert.view(-1, 768))

        other_idxs = torch.LongTensor(idxs_sent_other)[torch.randperm(len(idxs_sent_other))]
        support_other_bert, support_other_labels = bert_model.get_from_sent_idxs(other_idxs[:s2])
        support_other = model(support_other_bert.view(-1, 768))
        query_other_bert, query_other_labels = bert_model.get_from_sent_idxs(other_idxs[s2:s2 + int(s1 * negative_sampling_ratio)])
        query_other = model(query_other_bert.view(-1, 768))

        support = torch.cat((support_min, support_other), dim=0)
        support_labels = torch.cat((support_min_labels, support_other_labels), dim=0)

        input = torch.cat((query_min, query_other), dim=0)
        input_labels = torch.cat((query_min_labels, query_other_labels), dim=0)

        idxs_keep_input = ((input_labels != 0) & (input_labels != num_labels - 2) & (input_labels != num_labels - 1)).view(-1)
        input = input[idxs_keep_input]
        labels = input_labels.view(-1)[idxs_keep_input] - 1

        distances = []
        centers = []
        for l in range(1, num_labels - 2):
            if l == 1:
                distances.append(model.bo.view(1, 1).expand(input.shape[0], 1))
            else:
                filter = (support_labels == l).view(-1)
                if filter.sum() == 0:
                    cent = None
                    dist = torch.tensor([-400.]).view(1, 1).expand(input.shape[0], 1).cuda()
                else:
                    cent = support[filter].mean(dim=0).view(1, -1)
                    dist = -((input - cent) ** 2).sum(dim=1).view(-1, 1)

                distances.append(dist)
                centers.append(cent)

        distances = torch.cat(distances, dim=1)
        loss = loss_f(distances, labels.cuda())

        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        bert.bert.zero_grad()
        torch.cuda.empty_cache()

        if running_mean:
            with torch.no_grad():
                for k, cent in enumerate(centers):
                    if cent is None:
                        continue

                    if all_centers[k] is None:
                        all_centers[k] = cent.detach().clone()
                    else:
                        all_centers[k] = alpha * cent.detach().clone() + (1 - alpha) * all_centers[k]

        if (epoch % 50) == 0:
            model.eval()
            bert.bert.eval()
            with torch.no_grad():
                log_all(distances.cpu(), labels.cpu(), "train", True, False, epoch * (len(idxs_sent_min) + s1 * negative_sampling_ratio - s1))

                if running_mean:
                    centers = all_centers
                else:
                    centers = []
                    for l in range(2, num_labels - 2):
                        sum = torch.zeros(model.embedding_dim).cuda()
                        total = 0
                        for tb in range(0, y.shape[0], 256):
                            ub = min(tb + 256, y.shape[0] - 1)
                            trembs, trlabs = bert_model.get_from_sent_idxs(range(tb, ub), train=True)
                            filter = (trlabs == l).view(-1)
                            if filter.sum() != 0:
                                sum += model(trembs.view(-1, 768)[filter].cuda()).sum(dim=0).view(-1)
                                total += filter.sum()

                        if total == 0:
                            cent = None
                        else:
                            cent = (sum / total).view(1, -1)
                        centers.append(cent)

                distances = torch.zeros(0, num_labels - 3)
                test_labs = torch.zeros(0, dtype=torch.long)
                all_true, all_pred = [], []
                for b in range(0, test_data["label_ids"].shape[0], 256):
                    ub = min(b + 256, test_data["label_ids"].shape[0] - 1)
                    test_embs, test_labels, valid_test_embs, original_test_labs = bert_model.get_from_sent_idxs(range(b, ub), train=False,
                                                                                                                output_valid=True)
                    query = model(test_embs.view(-1, 768).cuda())
                    idxs_keep_input = ((test_labels != 0) & (test_labels != num_labels - 2) & (test_labels != num_labels - 1)).view(-1)
                    query = query[idxs_keep_input]
                    t_labels = test_labels.view(-1)[idxs_keep_input] - 1

                    dists = []
                    dists.append(model.bo.view(1, 1).expand(query.shape[0], 1))
                    for cent in centers:
                        if cent is None:
                            dists.append(torch.tensor([-400.]).view(1, 1).expand(query.shape[0], 1).cuda())
                        else:
                            dist = -((query - cent) ** 2).sum(dim=1).view(-1, 1)
                            dists.append(dist)

                    dists = torch.cat(dists, dim=1)
                    distances = torch.cat((distances, dists.cpu()), dim=0)
                    test_labs = torch.cat((test_labs, t_labels.cpu()))

                    # ----------------------------------------------

                    query_2 = model(valid_test_embs.view(-1, 768).cuda())

                    dists_2 = []
                    dists_2.append(model.bo.view(1, 1).expand(query_2.shape[0], 1))
                    for cent in centers:
                        if cent is None:
                            dists_2.append(torch.tensor([-400.]).view(1, 1).expand(query_2.shape[0], 1).cuda())
                        else:
                            dist = -((query_2 - cent) ** 2).sum(dim=1).view(-1, 1)
                            dists_2.append(dist)

                    dists_2 = torch.cat(dists_2, dim=1).view(valid_test_embs.shape[0], valid_test_embs.shape[1], -1)
                    y_true, y_pred = get_labels(
                            torch.cat((torch.zeros(dists_2.shape[0], dists_2.shape[1], 1).cuda() - 500, dists_2), dim=2),
                            original_test_labs, bert.label_map)
                    all_true.extend(y_true)
                    all_pred.extend(y_pred)

                log_all(distances.cpu(), test_labs.cpu(), "test", True, False, epoch * (len(idxs_sent_min) + s1 * negative_sampling_ratio - s1))
                f1 = log_all(all_pred, all_true, "test", False, True, epoch * (len(idxs_sent_min) + s1 * negative_sampling_ratio - s1))

                if f1 > current_val_max:
                    current_val_max = f1
                    current_val_max_times = 0
                else:
                    current_val_max_times += 1

                del test_labs, query, t_labels, dists

        del min_idxs, support_min_bert, support_min_labels, support_min, query_min_bert, query_min_labels, query_min, \
            other_idxs, support_other_bert, support_other_labels, support_other, query_other_bert, query_other_labels, query_other, \
            support, support_labels, input, input_labels, idxs_keep_input, labels, distances, centers, loss
    return model


def train_standard(lr, epochs, bert_model):
    loss_f = nn.CrossEntropyLoss()

    size_train = bert_model.train_data["label_ids"].shape[0]
    size_test = bert_model.test_data["label_ids"].shape[0]

    model = StandardNetwork(768, num_labels - 3).cuda()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, bert_model, lr, warmup_proportion=warmup_proportion, total_steps=total_steps)

    for epoch in trange(epochs):
        permutation = torch.randperm(size_train)
        for i, b in enumerate(range(0, size_train, 64)):
            model.train()
            bert.bert.train()

            ub = min(b + 64, size_train)
            indices = permutation[b: ub]

            train_embs, train_labs = bert_model.get_from_sent_idxs(indices)
            output = model(train_embs.view(-1, 768))

            idxs_keep_output = ((train_labs != 0) & (train_labs != num_labels-2) & (train_labs != num_labels-1)).view(-1)
            output = output[idxs_keep_output]
            labels = train_labs.view(-1)[idxs_keep_output] - 1

            loss = loss_f(output, labels.cuda())

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            bert.bert.zero_grad()

            if (i % 25) == 0:
                model.eval()
                bert.bert.eval()
                with torch.no_grad():
                    log_all(output.cpu(), labels.cpu(), "train", True, False, size_train * epoch + 64 * i)

                    all_outputs = torch.zeros(0, num_labels - 3)
                    all_labels = torch.zeros(0, dtype=torch.long)
                    all_true, all_pred = [], []
                    for j, b in enumerate(range(0, size_test, 256)):
                        ub = min(b + 256, size_test)
                        test_embs, test_labs, valid_test_embs, original_test_labs = bert_model.get_from_sent_idxs(range(b, ub), train=False,
                                                                                                                  output_valid=True)
                        output = model(test_embs)
                        output2 = model(valid_test_embs)

                        y_true, y_pred = get_labels(torch.cat((torch.zeros(output2.shape[0], output2.shape[1], 1).cuda() - 500, output2), dim=2),
                                                    original_test_labs, bert.label_map)
                        all_true.extend(y_true)
                        all_pred.extend(y_pred)

                        idxs_keep_output = ((test_labs != 0) & (test_labs != num_labels - 2) & (test_labs != num_labels - 1)).view(-1)
                        output = output[idxs_keep_output]
                        labels = test_labs.view(-1)[idxs_keep_output] - 1

                        all_labels = torch.cat((all_labels, labels.cpu()))
                        all_outputs = torch.cat((all_outputs, output.cpu()))

                    log_all(all_outputs.cpu(), all_labels.cpu(), "test", True, False, size_train * epoch + 64 * i)
                    log_all(all_pred, all_true, "test", False, True, size_train * epoch + 64 * i)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments with protonetwork and standard network')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--proto', action='store_true')
    parser.add_argument('--bert', type=str, default="True", choices=["True", "Mock", "true", "mock"])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--negative-sampling', type=float, default=2)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--steps', type=float, default=756.)
    parser.add_argument('--dataset', type=str, default="jnlpba-25dna")
    parser.add_argument('--emb-dim', type=int, default=300)
    parser.add_argument('--reduced-labels', type=int, default=[2, 3], nargs=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--support-min', type=int, default=10)
    parser.add_argument('--support-other', type=int, default=100)
    args = parser.parse_args()
    print(args)
    print(torch.cuda.memory_allocated())

    use_wandb = args.wandb
    proto = args.proto
    bert_type = args.bert
    lr = args.lr
    negative_sample_ratio = args.negative_sampling
    warmup_proportion = args.warmup
    total_steps = args.steps
    dataset = args.dataset
    embedding_dimension = args.emb_dim
    reduced_labels = tuple(args.reduced_labels)
    support_min = args.support_min
    support_other = args.support_other

    num_labels = len(get_processors()[dataset]().get_labels()) + 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if use_wandb:
        run = wandb.init(name="", config={
                "proto":                 proto,
                "bert_type":             bert_type,
                "lr":                    lr,
                "negative_sample_ratio": negative_sample_ratio,
                "warmup_proportion":     warmup_proportion,
                "total_steps":           total_steps,
                "dataset":               dataset,
                "embedding_dimension":   embedding_dimension,
                "reduced_labels":        reduced_labels,
                "support_min":           support_min,
                "support_other":         support_other,
                "seed":                  args.seed
        }, project="proto-memory")
        wandb.save("*.py")

    if bert_type.lower() == "true":
        bert = TrueBert(dataset)
        print(torch.cuda.memory_allocated())
    elif bert_type.lower() == "mock":
        bert = MockTrainedBert(dataset)
    else:
        raise ValueError

    if proto:
        train_proto(
                bert.train_data["label_ids"],
                bert.test_data,
                bert,
                epochs=int(total_steps),
                negative_sampling_ratio=negative_sample_ratio,
                lr=lr,
                l1=reduced_labels[0],
                l2=reduced_labels[1],
                support_min=support_min,
                support_other=support_other,
                embedding_dimension=embedding_dimension
        )
    else:
        train_standard(lr=lr, epochs=4, bert_model=bert)
