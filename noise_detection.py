import torch
from tqdm import tqdm, trange

from sklearn.metrics import auc
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

from sklearn.cluster import KMeans

from .consts import get_base_parameters_trainer
from .trainer import Trainer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Model trainer')
    for k, v in get_base_parameters_trainer().items():
        if type(v) == bool:
            parser.add_argument(f"--{k.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)

    args = dict(vars(parser.parse_args()))

    dataset_name = args["dataset_name"]
    noise = args["noise_addition"]
    model_name = args["model_name"]
    epochs = args["num_train_epochs"]

    pars = get_base_parameters_trainer()
    pars["wandb"] = False
    pars["dataset_name"] = dataset_name
    pars["task_name"] = "ner"
    pars["model_name"] = model_name
    pars["max_seq_length"] = 50
    pars["num_train_epochs"] = epochs
    pars["do_eval"] = False
    pars["noise_addition"] = noise

    t = Trainer(**pars)
    t.train()

    train_losses = torch.zeros(t.train_dataloader.dataset.tensors[0].view(-1).shape[0])
    train_noises = t.train_dataloader.dataset.tensors[6]
    t.eval_on = 'test'
    t.test_examples = t.processor.get_test_examples(t.data_dir)
    test_losses = torch.zeros(t.get_dataloader(train=False, force_recompute=True, label_noise_addition=noise).dataset.tensors[0].view(-1).shape[0])
    test_noises = t.test_dataloader.dataset.tensors[6]
    train_labels = t.train_dataloader.dataset.tensors[3]
    test_labels = t.test_dataloader.dataset.tensors[3]
    train_max_logits = torch.zeros(t.train_dataloader.dataset.tensors[0].view(-1).shape[0])
    test_max_logits = torch.zeros(t.get_dataloader(train=False, force_recompute=True, label_noise_addition=noise).dataset.tensors[0].view(-1).shape[0])

    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, noise_mask, selected_idxs in tqdm(t.train_dataloader):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            valid_ids = valid_ids.cuda()
            label_ids = label_ids.cuda()
            l_mask = l_mask.cuda()
            noise_mask = noise_mask.cuda()
            selected_idxs = selected_idxs.cuda()

            _, logits = t.model(input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                valid_ids=valid_ids,
                                examples_indexes=selected_idxs,
                                task=t.eval_on,
                                step=0,
                                nth_layer=12,
                                labels=label_ids)


            train_losses[selected_idxs.view(-1)] = t.model.last_losses.cpu()
            train_max_logits[selected_idxs.view(-1)] = logits.max(-1).values.view(-1).cpu()

        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, noise_mask, selected_idxs in tqdm(t.test_dataloader):
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            valid_ids = valid_ids.cuda()
            label_ids = label_ids.cuda()
            l_mask = l_mask.cuda()
            noise_mask = noise_mask.cuda()
            selected_idxs = selected_idxs.cuda()

            _, logits = t.model(input_ids,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask,
                                valid_ids=valid_ids,
                                examples_indexes=selected_idxs,
                                task=t.eval_on,
                                step=0,
                                nth_layer=12,
                                labels=label_ids)

            test_losses[selected_idxs.view(-1)] = t.model.last_losses.cpu()
            test_max_logits[selected_idxs.view(-1)] = logits.max(-1).values.view(-1).cpu()

    train_noises = train_noises.view(-1)
    test_noises = test_noises.view(-1)

    original_train_losses = train_losses.clone()
    idxs = torch.randperm(train_losses.shape[0])
    cutoff = int(idxs.shape[0] * 0.9)
    train_idxs = idxs[:cutoff]
    test_idxs = idxs[cutoff:]
    train_losses, test_losses = train_losses[train_idxs], train_losses[test_idxs]
    train_noises, test_noises = train_noises[train_idxs], train_noises[test_idxs]
    train_max_logits, test_max_logits = train_max_logits[train_idxs], train_max_logits[test_idxs]

    sorted_train_losses = train_losses.sort(dim=0, descending=True)
    sorted_test_losses = test_losses.sort(dim=0, descending=True)
    sorted_train_noises = train_noises[sorted_train_losses.indices]
    sorted_test_noises = test_noises[sorted_test_losses.indices]
    sorted_train_max_logits = train_max_logits[sorted_train_losses.indices]
    sorted_test_max_logits = test_max_logits[sorted_test_losses.indices]

    precisions = []
    recalls = []
    f1s = []
    threshs = []
    for i in trange(1, sorted_test_noises.shape[0], 10):
        true = sorted_test_noises[:i]
        pred = torch.ones_like(true)
        TP = ((true == 1) & (pred == 1)).sum().float()
        FP = ((true == 0) & (pred == 1)).sum().float()
        FN = ((true == 1) & (pred == 0)).sum().float()

        if TP + FP == 0:
            pr = 0.
        else:
            pr = (TP / (TP + FP)).item()
        precisions.append(pr)

        if TP + FN == 0:
            re = 0.
        else:
            re = (TP / sorted_test_noises.sum()).item()
        recalls.append(re)

        if pr + re == 0:
            f1 = 0.
        else:
            f1 = 2 * pr * re / (pr + re)
        f1s.append(f1)
        threshs.append(sorted_test_losses.values[i])

    smoothed_ap = []
    for i, (p, r) in enumerate(tqdm(zip(precisions, recalls), total=len(precisions))):
        smoothed_ap.append(max(precisions[i:]))

    argmax_f1 = max(zip(f1s, range(len(f1s))))[1]
    max_f1 = f1s[argmax_f1]
    max_f1_thresh = threshs[argmax_f1]

    plt.figure(figsize=(8, 6))
    plt.rc('font', family='serif')
    plt.rc('ytick', labelsize='x-large')
    plt.rc('xtick', labelsize='x-large')
    plt.grid(True)
    plt.plot(recalls, smoothed_ap, color="k")
    plt.title(f"AUC AP: {auc(recalls, smoothed_ap):.4f} - max F1: {max_f1:.4f}({max_f1_thresh:.4f})", fontsize=18)
    plt.legend(["AP"], loc=3, fontsize=14)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.savefig(f'AP_test_{dataset_name}_{model_name.replace("-", "").replace("/", "")}_{int(noise * 100)}.pdf', format='pdf')

    km = KMeans(n_clusters=2)
    km.fit(train_losses.view(-1, 1))
    km_pred = km.predict(test_losses.view(-1, 1))
    km_pred = torch.tensor(km_pred)

    linspace = torch.linspace(train_losses.min(), train_losses.max(), steps=1000)
    km_linspace = km.predict(linspace.view(-1, 1))
    v = 0
    if km_linspace.nonzero()[0][0] != 0:
        v = linspace[km_linspace.nonzero()[0][0]]
    else:
        v = linspace[km_linspace.nonzero()[0][-1]]

    blue = torch.tensor([0 / 255, 42 / 255, 92 / 255]) * 0.4 + torch.tensor([66 / 255, 148 / 255, 255 / 255]) * 0.6
    red = torch.tensor([112 / 255, 24 / 255, 0 / 255]) * 0.4 + torch.tensor([222 / 255, 100 / 255, 67 / 255]) * 0.6
    fig = plt.figure(figsize=(8, 6))
    bax = brokenaxes(
        ylims=((0, 60000), ((train_losses < 0.18).sum().item() - 15000, (train_losses < 0.2).sum().item() + 15000)),
        wspace=.1, despine=False, fig=fig)
    plt.rc('ytick', labelsize='x-large')
    plt.rc('xtick', labelsize='x-large')
    normal = bax.hist(train_losses[(train_noises == 0)], bins=25, alpha=0)
    noisy = bax.hist(train_losses[(train_noises == 1)], bins=25, alpha=0)
    bax.set_axisbelow(True)
    bax.hist(train_losses[(train_noises == 0) & (train_losses <= v)], hatch="///", edgecolor="white", label="Normal",
             facecolor=blue, bins=normal[0][1], lw=0)
    bax.hist(train_losses[(train_noises == 1) & (train_losses > v)], hatch="\\\\\\", edgecolor="white", label="Noisy",
             facecolor=red, bins=noisy[0][1], lw=0)
    bax.hist(train_losses[(train_noises == 0) & (train_losses > v)], hatch="///", edgecolor="white", facecolor=blue,
             bins=normal[0][1], lw=0)
    bax.hist(train_losses[(train_noises == 1) & (train_losses <= v)], hatch="\\\\\\", edgecolor="white", facecolor=red,
             bins=noisy[0][1], lw=0)
    bax.grid(True)
    bax.hist(train_losses[train_noises == 0], histtype="step", lw=1.5, edgecolor="black", bins=normal[0][1])
    bax.hist(train_losses[train_noises == 1], histtype="step", lw=1.5, edgecolor="black", bins=noisy[0][1])
    bax.axvline(v, ls="--", c="gray", lw=1, label="Classifier cutoff")
    bax.legend(loc=1, fontsize=14)
    bax.set_xlabel('Training loss', labelpad=20, fontsize=16)
    bax.set_ylabel('Number of examples', labelpad=70, fontsize=16)
    plt.rc('font', family='serif')
    fig.savefig(f'histogram_losses_{dataset_name}_{model_name.replace("-", "").replace("/", "")}_{int(noise * 100)}.pdf', format='pdf', bbox_inches='tight')

    TP = ((km_pred == 1) & (test_noises == 1)).sum().float().item()
    FP = ((km_pred == 1) & (test_noises == 0)).sum().float().item()
    FN = ((km_pred == 0) & (test_noises == 1)).sum().float().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1s = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    TP2 = ((km_pred == 0) & (test_noises == 1)).sum().float().item()
    FP2 = ((km_pred == 0) & (test_noises == 0)).sum().float().item()
    FN2 = ((km_pred == 1) & (test_noises == 1)).sum().float().item()

    precision2 = TP2 / (TP2 + FP2) if TP2 + FP2 > 0 else 0
    recall2 = TP2 / (TP2 + FN2) if TP2 + FN2 > 0 else 0
    f1s2 = 2 * precision2 * recall2 / (precision2 + recall2) if precision2 + recall2 > 0 else 0

    with open(f'scores_{dataset_name}_{model_name.replace("-", "").replace("/", "")}_{int(noise * 100)}.txt', "w") as f:
        f.writelines([f"{precision}\t{recall}\t{f1s}\n",
                      f"{precision2}\t{recall2}\t{f1s2}"])
