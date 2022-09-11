import torch

from .seqeval_modified import f1_score, accuracy_score, recall_score, precision_score, classification_report
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score
from sklearn.metrics import fbeta_score as sk_fbeta_score
from .wnuteval import doc_to_toks, doc_to_entities, fmt_results, get_tagged_entities, calc_results, get_tags, filter_entities


NER_LABELS = ('O', 'I-LOC', 'I-PER', 'I-MISC', 'I-ORG', 'B-LOC', 'B-PER', 'B-MISC', 'B-ORG')


def get_TP(pred, real, positive=1, ignore=None):
    if ignore is None:
        return ((pred == positive) & (real == positive)).int().sum()
    else:
        return ((pred == positive) & (real == positive) &  (real != ignore)).int().sum()


def get_TN(pred, real, positive=1, ignore=None):
    if ignore is None:
        return ((pred == 0) & (real == 0)).int().sum()
    else:
        return ((pred == 0) & (real == 0) & (real != ignore)).int().sum()


def get_FP(pred, real, positive=1, ignore=None):
    if ignore is None:
        return ((pred == positive) & (real == 0)).int().sum()
    else:
        return ((pred == positive) & (real == 0) & (real != ignore)).int().sum()


def get_FN(pred, real, positive=1, ignore=None):
    if ignore is None:
        return ((pred == 0) & (real == positive)).int().sum()
    else:
        return ((pred == 0) & (real == positive) & (real != ignore)).int().sum()


def classify(data, multiclass=True):
    if multiclass:
        classif = data.argmax(dim=1)
    else:
        classif = (data >= 0.5).long()
    return classif


def get_accuracy(pred, real, positive=1):
    pred = classify(pred)

    same = (pred == real).view(-1).int().sum()
    return same.float() / pred.view(-1).shape[0]


def get_precision(pred, real, positive=1):
    pred = classify(pred)

    TP = ((pred == positive) & (real == positive)).int().sum()
    FP = ((pred == positive) & (real == 0)).int().sum()

    if (TP + FP).item() == 0:
        return torch.tensor([0.])

    return TP.float() / (TP + FP)


def get_recall(pred, real, positive=1):
    pred = classify(pred)

    TP = ((pred == positive) & (real == positive)).int().sum()
    FN = ((pred == 0) & (real == positive)).int().sum()

    if (TP + FN).item() == 0:
        return torch.tensor([0.])

    return TP.float() / (TP + FN)


def get_f05(precision, recall, positive=1):
    if precision == 0 and recall == 0:
        return 0.
    return (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)


def get_ner_metrics(y_true, y_pred, digits=4, average="micro", skipreport=False):
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)

    if precision + recall == 0:
        f05score = 0
    else:
        f05score = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)
    return {
        "report": None if skipreport else classification_report(y_true, y_pred, digits=digits),
        "precision": precision,
        "f1score": f1_score(y_true, y_pred, average=average),
        "recall": recall,
        "accuracy": accuracy_score(y_true, y_pred),
        "f05score": f05score
    }


def get_wnut_metrics(y_true, y_pred, digits=4, average="micro"):
    lines = []
    for s in range(len(y_true)):
        line = []
        for t in range(len(y_true[s])):
            pred = y_pred[s][t]
            if pred == "[CLS]" or pred == "[SEP]" or pred == "[PAD]":
                pred = "O"
            line.append(f"PLACEHOLDER\t{y_true[s][t]}\t{pred}")
        lines.extend(line)
        lines.append("")

    tokens = doc_to_toks(lines)
    all_entities = doc_to_entities(lines)

    # report results
    _sys = 'sys_1'
    # throw out 'O' tags to get overall p/r/f
    tagged_entities = get_tagged_entities(all_entities)
    results = {'all':    calc_results(all_entities['gold'], all_entities[_sys], surface_form=False),
               'tagged': calc_results(tagged_entities['gold'], tagged_entities[_sys], False),
               'tokens': calc_results(tokens['gold'], tokens[_sys], surface_form=False)}

    accuracy = results['tokens'].correct / results['tokens'].gold
    precision = results['tagged'].p
    recall = results['tagged'].r
    f1score = results['tagged'].f

    tags = get_tags(all_entities['gold'])
    lines = ["precision\trecall\tf1_score\tsupport\n"]
    for tag in sorted(tags):
        ents = {src: filter_entities(entities, lambda e: e.tag == tag)
                    for src, entities in all_entities.items()}
        results = calc_results(ents['gold'], ents[_sys], False)
        lines.append(f"{tag}\t{results.p}\t{results.r}\t{results.f}\t0")

    lines += [""]
    lines += [f"micro\t{precision}\t{recall}\t{f1score}\t0"]
    lines += [f"macro\t{precision}\t{recall}\t{f1score}\t0"]
    report = "\n".join(lines)

    return {
            "report": report,
            "precision": precision,
            "f1score": f1score,
            "recall": recall,
            "accuracy": accuracy,
            "f05score": f1score
    }


def get_nli_metrics(y_true, y_pred, digits=4):
    y_true = y_true[0]
    y_pred = y_pred[0]
    TPs = [0., 0., 0.]
    FPs = [0., 0., 0.]
    FNs = [0., 0., 0.]
    TNs = [0., 0., 0.]
    labels = ["neutral", "entailment", "contradiction"]
    other = 0.

    for s_true, s_pred in zip(y_true, y_pred):
        for i, l in enumerate(labels):
            if s_true == s_pred == labels[i]:
                TPs[i] += 1
            elif s_true == s_pred != labels[i]:
                TNs[i] += 1
            elif s_true == labels[i] and s_pred != labels[i]:
                FNs[i] += 1
            elif s_true != labels[i] and s_pred == labels[i]:
                FPs[i] += 1
            else:
                other += 1

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    f05_scores = []
    for i in range(len(labels)):
        if (TPs[i] + FPs[i] + FNs[i] + TNs[i]) == 0:
            accuracy = 0.
        else:
            accuracy = (TPs[i] + TNs[i]) / (TPs[i] + FPs[i] + FNs[i] + TNs[i])

        if (TPs[i] + FPs[i]) == 0:
            precision = 0.
        else:
            precision = TPs[i] / (TPs[i] + FPs[i])

        if (TPs[i] + FNs[i]) == 0:
            recall = 0.
        else:
            recall = TPs[i] / (TPs[i] + FNs[i])

        if precision == 0 and recall == 0:
            f05 = 0
            f1s = 0
        else:
            beta = 0.5
            f05 = (1 + beta ** 2) * TPs[i] / ((1 + beta ** 2) * TPs[i] + beta ** 2 * FNs[i] + FPs[i])
            beta = 1.
            f1s = (1 + beta ** 2) * TPs[i] / ((1 + beta ** 2) * TPs[i] + beta ** 2 * FNs[i] + FPs[i])

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1s)
        f05_scores.append(f05)

    to_write = [
        ["accuracy:", f"{accuracies[0]:.{digits}f}\t{accuracies[1]:.{digits}f}\t{accuracies[2]:.{digits}f}"],
        ["precision:", f"{precisions[0]:.{digits}f}\t{precisions[1]:.{digits}f}\t{precisions[2]:.{digits}f}"],
        ["recall:", f"{recalls[0]:.{digits}f}\t{recalls[1]:.{digits}f}\t{recalls[2]:.{digits}f}"],
        ["f05 score:", f"{f05_scores[0]:.{digits}f}\t{f05_scores[1]:.{digits}f}\t{f05_scores[2]:.{digits}f}\t"]
    ]

    col_width = max(len(word) for row in to_write for word in row) + 2  # padding

    report = "\n".join(["".join(word.rjust(col_width) for word in row) for row in to_write])

    return {
        "report": report,
        "precision": sum(precisions)/len(labels),
        "f1score": sum(f1_scores)/len(labels),
        "f05score": sum(f05_scores)/len(labels),
        "recall": sum(recalls)/len(labels),
        "accuracy": sum(accuracies)/len(labels),
        "TP": sum(TPs),
        "FP": sum(FPs),
        "FN": sum(FNs),
        "TN": sum(TNs),
        "other": other
    }


def get_ir_metrics(y_true, y_pred, digits=4):
    y_true = y_true[0]
    y_pred = y_pred[0]

    accuracy = sk_accuracy_score(y_true, y_pred)
    precision = sk_precision_score(y_true, y_pred, average="micro")
    recall = sk_recall_score(y_true, y_pred, average="micro")
    f1_score = sk_fbeta_score(y_true, y_pred, beta=1., average="micro")
    f05_score = sk_fbeta_score(y_true, y_pred, beta=.5, average="micro")

    to_write = [
        ["accuracy:", f"{accuracy:.{digits}f}"],
        ["precision:", f"{precision:.{digits}f}"],
        ["recall:", f"{recall:.{digits}f}"],
        ["f05 score:", f"{f05_score:.{digits}f}"],
        ["f1 score:", f"{f1_score:.{digits}f}"],
    ]

    col_width = max(len(word) for row in to_write for word in row) + 2  # padding

    report = "\n".join(["".join(word.rjust(col_width) for word in row) for row in to_write])

    return {
        "report": report,
        "precision": precision,
        "f1score": f1_score,
        "f05score": f05_score,
        "recall": recall,
        "accuracy": accuracy,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "other": 0
    }


def get_ged_metrics(y_true, y_pred, digits=4):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    other = 0

    for s_true, s_pred in zip(y_true, y_pred):
        for w_true, w_pred in zip(s_true, s_pred):
            if w_true == w_pred == 'c':
                TN += 1
            elif w_true == w_pred == 'i':
                TP += 1
            elif w_true == 'c' and w_pred == 'i':
                FP += 1
            elif w_true == 'i' and w_pred == 'c':
                FN += 1
            else:
                other += 1

    TP, FP, FN, TN = float(TP), float(FP), float(FN), float(TN)

    if (TP + FP + FN + TN) == 0:
        accuracy = 0.
    else:
        accuracy = (TP + TN) / (TP + FP + FN + TN)

    if (TP + FP) == 0:
        precision = 0.
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 0.
    else:
        recall = TP / (TP + FN)

    if precision == 0 and recall == 0:
        f05 = 0
        f1s = 0
    else:
        beta = 0.5
        f05 = (1 + beta ** 2) * TP / ((1 + beta ** 2) * TP + beta ** 2 * FN + FP)
        beta = 1.
        f1s = (1 + beta ** 2) * TP / ((1 + beta ** 2) * TP + beta ** 2 * FN + FP)

    to_write = [
        ["accuracy:", f"{accuracy:.{digits}f}"],
        ["precision:", f"{precision:.{digits}f}"],
        ["recall:", f"{recall:.{digits}f}"],
        ["f05 score:", f"{f05:.{digits}f}"]
    ]

    col_width = max(len(word) for row in to_write for word in row) + 2  # padding

    report = "\n".join(["".join(word.rjust(col_width) for word in row) for row in to_write])

    return {
        "report": report,
        "precision": precision,
        "f1score": f1s,
        "f05score": f05,
        "recall": recall,
        "accuracy": accuracy,
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
        "other": other
    }


def get_all_metrics(pred, real, positive=1, multiclass=True, ner=False):
    if ner and multiclass:
        real = [NER_LABELS[i] for i in real]
        pred = [NER_LABELS[i] for i in pred]

        precision = precision_score(real, pred)
        recall = recall_score(real, pred)
        accuracy = accuracy_score(real, pred)
        f1s = f1_score(real, pred)

        return {
            "accuracy": (accuracy, accuracy * 100.),
            "precision": (precision, precision * 100.),
            "recall": (recall, recall * 100.),
            "f1s": (f1s, f1s * 100.)
        }
    else:
        TP = get_TP(pred, real, positive=positive)
        FN = get_FN(pred, real, positive=positive)
        FP = get_FP(pred, real, positive=positive)
        TN = get_TN(pred, real, positive=positive)

        accuracy = ((TP + TN).float() / (TP + FP + FN + TN)).item()

        if (TP + FP).item() == 0:
            precision = 0.
        else:
            precision = (TP.float() / (TP + FP)).item()

        if (TP + FN).item() == 0:
            recall = 0.
        else:
            recall = (TP.float() / (TP + FN)).item()

        if precision == 0 and recall == 0:
            f05 = 0
            f1s = 0
        else:
            f05 = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)
            f1s = 2 * precision * recall / (precision + recall)

        return {
            "TP": int(TP.item()),
            "FN": int(FN.item()),
            "FP": int(FP.item()),
            "TN": int(TN.item()),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f05score": f05,
            "f1score": f1s
        }
