import torch
from transformers import BertPreTrainedModel, BertModel, RobertaModel, DebertaPreTrainedModel, DebertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch import nn
from torch.nn import functional as F
import wandb


class BertTokenClassifier(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super(BertTokenClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.wandb = config.wandb

        self.bert = BertModel(config, add_pooling_layer=False)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()

        self.last_losses = None
        self.last_active_loss = None
        self.last_logits = None
        self.last_labels = None
        self.last_used_idxs = None

        self.epoch = None

    def verify_noise_detection(self, noise_mask, step=-1):
        if noise_mask is not None and noise_mask.sum().item() != 0:
            noise_mask = noise_mask.view(-1)[self.last_active_loss]

            sorter = self.last_losses
            k = noise_mask.sum()

            top_k_sorter = torch.topk(sorter, k)[1]
            sorter[top_k_sorter] = 1
            sorter[sorter != 1] = 0

            TP = ((noise_mask == 1) & (sorter == 1)).int().sum()
            FP = ((noise_mask == 0) & (sorter == 1)).int().sum()
            FN = ((noise_mask == 1) & (sorter == 0)).int().sum()
            TN = ((noise_mask == 0) & (sorter == 0)).int().sum()
            tot = noise_mask.view(-1).shape[0]

            print(f"noise detection: TP: {TP}\tFP: {FP}\tFN: {FN}\tTN: {TN}\ttot: {tot}")

            noise_mask = noise_mask.bool()
            predicted_labels_noise = self.last_logits[noise_mask].argmax(-1)
            accuracy_noise = (predicted_labels_noise == self.last_labels[noise_mask]).float().sum().item() / k.item()

            if self.wandb:
                wandb.log({
                    "noise/classification-accuracy": accuracy_noise,
                    "noise/detection-accuracy": (TP + TN).float()/(TP + TN + FP + FN),
                    "noise/detection-f1score": (2 * TP).float() / (2 * TP + FN + FP)
                }, step=step)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, **kwargs):

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)
        sequence_output = bert_output[0]
        try:
            n = kwargs["nth_layer"]
        except KeyError:
            n = 12
        nth_layer_output = bert_output[-1][n]
        if n != 12: print(f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} hidden layer used")

        if torch.isnan(sequence_output).any():
            raise ValueError("NaNs in sequence output")

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        valid_output_nth_layer = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        last_used_idxs = torch.zeros(batch_size, max_len, dtype=torch.long).cuda() - 1
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
                    valid_output_nth_layer[i][jj] = nth_layer_output[i][j]
                    last_used_idxs[i][j] = kwargs["examples_indexes"][i][j].item()

        logits = F.softmax(self.classifier(valid_output), dim=-1)

        del sequence_output, valid_output
        sequence_output = nth_layer_output

        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=0, reduction="none")

            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]

                active_labels = labels.view(-1)[active_loss]

                self.last_used_idxs = last_used_idxs.view(-1)[active_loss]

                loss_expanded = loss_fct(torch.log(active_logits), active_labels)

                self.last_labels = labels.detach().clone().view(-1)[active_loss]
                self.last_logits = logits.detach().clone().view(-1, self.num_labels)[active_loss]

                loss = loss_expanded.mean()
            else:
                active_loss = torch.zeros(labels.shape[0]) == 0
                active_labels = labels
                loss_expanded = loss_fct(torch.log(logits.view(-1, self.num_labels)), active_labels.view(-1))
                loss = loss_expanded.mean()

            with torch.no_grad():
                self.last_losses = loss_expanded.detach().view(-1).clone()
                self.last_active_loss = active_loss.detach().clone().view(-1)

                del loss_expanded

            return loss, logits
        else:
            return logits, sequence_output


class RobertaTokenClassifier(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super(RobertaTokenClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.wandb = config.wandb

        self.roberta = RobertaModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()

        self.last_losses = None
        self.last_active_loss = None
        self.last_logits = None
        self.last_labels = None
        self.last_used_idxs = None

        self.epoch = None

    def verify_noise_detection(self, noise_mask, step=-1):
        if noise_mask is not None and noise_mask.sum().item() != 0:
            noise_mask = noise_mask.view(-1)[self.last_active_loss]

            sorter = self.last_losses
            k = noise_mask.sum()

            top_k_sorter = torch.topk(sorter, k)[1]
            sorter[top_k_sorter] = 1
            sorter[sorter != 1] = 0

            TP = ((noise_mask == 1) & (sorter == 1)).int().sum()
            FP = ((noise_mask == 0) & (sorter == 1)).int().sum()
            FN = ((noise_mask == 1) & (sorter == 0)).int().sum()
            TN = ((noise_mask == 0) & (sorter == 0)).int().sum()
            tot = noise_mask.view(-1).shape[0]

            print(f"noise detection: TP: {TP}\tFP: {FP}\tFN: {FN}\tTN: {TN}\ttot: {tot}")

            noise_mask = noise_mask.bool()
            predicted_labels_noise = self.last_logits[noise_mask].argmax(-1)
            accuracy_noise = (predicted_labels_noise == self.last_labels[noise_mask]).float().sum().item() / k.item()

            if self.wandb:
                wandb.log({
                    "noise/classification-accuracy": accuracy_noise,
                    "noise/detection-accuracy": (TP + TN).float()/(TP + TN + FP + FN),
                    "noise/detection-f1score": (2 * TP).float() / (2 * TP + FN + FP)
                }, step=step)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, **kwargs):

        bert_output = self.roberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)
        sequence_output = bert_output[0]
        try:
            n = kwargs["nth_layer"]
        except KeyError:
            n = 12
        nth_layer_output = bert_output[-1][n]
        if n != 12: print(f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} hidden layer used")

        if torch.isnan(sequence_output).any():
            raise ValueError("NaNs in sequence output")

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        valid_output_nth_layer = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        last_used_idxs = torch.zeros(batch_size, max_len, dtype=torch.long).cuda() - 1
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
                    valid_output_nth_layer[i][jj] = nth_layer_output[i][j]
                    last_used_idxs[i][j] = kwargs["examples_indexes"][i][j].item()

        logits = F.softmax(self.classifier(valid_output), dim=-1)

        del sequence_output, valid_output
        sequence_output = nth_layer_output

        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=0, reduction="none")

            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]

                active_labels = labels.view(-1)[active_loss]

                self.last_used_idxs = last_used_idxs.view(-1)[active_loss]

                loss_expanded = loss_fct(torch.log(active_logits), active_labels)

                self.last_labels = labels.detach().clone().view(-1)[active_loss]
                self.last_logits = logits.detach().clone().view(-1, self.num_labels)[active_loss]

                loss = loss_expanded.mean()
            else:
                active_loss = torch.zeros(labels.shape[0]) == 0
                active_labels = labels
                loss_expanded = loss_fct(torch.log(logits.view(-1, self.num_labels)), active_labels.view(-1))
                loss = loss_expanded.mean()

            with torch.no_grad():
                self.last_losses = loss_expanded.detach().view(-1).clone()
                self.last_active_loss = active_loss.detach().clone().view(-1)

                del loss_expanded

            return loss, logits
        else:
            return logits, sequence_output


class DebertaTokenClassifier(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super(DebertaTokenClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.wandb = config.wandb

        self.deberta = DebertaModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.init_weights()

        self.last_losses = None
        self.last_active_loss = None
        self.last_logits = None
        self.last_labels = None
        self.last_used_idxs = None

        self.epoch = None

    def verify_noise_detection(self, noise_mask, step=-1):
        if noise_mask is not None and noise_mask.sum().item() != 0:
            noise_mask = noise_mask.view(-1)[self.last_active_loss]

            sorter = self.last_losses
            k = noise_mask.sum()

            top_k_sorter = torch.topk(sorter, k)[1]
            sorter[top_k_sorter] = 1
            sorter[sorter != 1] = 0

            TP = ((noise_mask == 1) & (sorter == 1)).int().sum()
            FP = ((noise_mask == 0) & (sorter == 1)).int().sum()
            FN = ((noise_mask == 1) & (sorter == 0)).int().sum()
            TN = ((noise_mask == 0) & (sorter == 0)).int().sum()
            tot = noise_mask.view(-1).shape[0]

            print(f"noise detection: TP: {TP}\tFP: {FP}\tFN: {FN}\tTN: {TN}\ttot: {tot}")

            noise_mask = noise_mask.bool()
            predicted_labels_noise = self.last_logits[noise_mask].argmax(-1)
            accuracy_noise = (predicted_labels_noise == self.last_labels[noise_mask]).float().sum().item() / k.item()

            if self.wandb:
                wandb.log({
                    "noise/classification-accuracy": accuracy_noise,
                    "noise/detection-accuracy": (TP + TN).float()/(TP + TN + FP + FN),
                    "noise/detection-f1score": (2 * TP).float() / (2 * TP + FN + FP)
                }, step=step)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, **kwargs):

        bert_output = self.deberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = bert_output[0]
        try:
            n = kwargs["nth_layer"]
        except KeyError:
            n = 12
        nth_layer_output = bert_output[-1][n]
        if n != 12: print(f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} hidden layer used")

        if torch.isnan(sequence_output).any():
            raise ValueError("NaNs in sequence output")

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        valid_output_nth_layer = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).cuda()
        last_used_idxs = torch.zeros(batch_size, max_len, dtype=torch.long).cuda() - 1
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
                    valid_output_nth_layer[i][jj] = nth_layer_output[i][j]
                    last_used_idxs[i][j] = kwargs["examples_indexes"][i][j].item()


        logits = F.softmax(self.classifier(valid_output), dim=-1)

        del sequence_output, valid_output
        sequence_output = nth_layer_output

        if labels is not None:
            loss_fct = nn.NLLLoss(ignore_index=0, reduction="none")

            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]

                active_labels = labels.view(-1)[active_loss]

                self.last_used_idxs = last_used_idxs.view(-1)[active_loss]

                loss_expanded = loss_fct(torch.log(active_logits), active_labels)

                self.last_labels = labels.detach().clone().view(-1)[active_loss]
                self.last_logits = logits.detach().clone().view(-1, self.num_labels)[active_loss]

                loss = loss_expanded.mean()
            else:
                active_loss = torch.zeros(labels.shape[0]) == 0
                active_labels = labels
                loss_expanded = loss_fct(torch.log(logits.view(-1, self.num_labels)), active_labels.view(-1))
                loss = loss_expanded.mean()

            with torch.no_grad():
                self.last_losses = loss_expanded.detach().view(-1).clone()
                self.last_active_loss = active_loss.detach().clone().view(-1)

                del loss_expanded

            return loss, logits
        else:
            return logits, sequence_output