import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101, resnet50, resnet152, wide_resnet101_2, resnext101_32x8d, resnext50_32x4d
import wandb


class MockTokenizer():
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None


class ResNetConfig:
    def __init__(self, name, hidden_dropout_prob, hidden_size, num_labels):
        self.name = name
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.memory_outputs_logits = True
        self.start_mem = 3
        self.start_adding = 2
        self.stop_adding = 5
        self.memory = None
        self.wandb = False


    @staticmethod
    def from_pretrained(name="", hidden_dropout_prob=0.1, hidden_size=1000, num_labels=10, **kwargs):
        return ResNetConfig(name, hidden_dropout_prob, hidden_size, num_labels)


class ResNet101Classifier(nn.Module):
    @staticmethod
    def from_pretrained(model_name, config=ResNetConfig.from_pretrained(), **kwargs):
        config.model_name = model_name
        return ResNet101Classifier(config=config)

    def __init__(self, config):
        super(ResNet101Classifier, self).__init__()
        self.num_labels = config.num_labels

        if config.model_name.lower() == "resnet101":
            self.resnet = resnext101_32x8d(num_classes=config.hidden_size, pretrained=True)
        elif config.model_name.lower() == "resnet50":
            self.resnet = resnext50_32x4d(num_classes=config.hidden_size, pretrained=True)
        elif config.model_name.lower() == "resnet152":
            self.resnet = resnet152(num_classes=config.hidden_size, pretrained=True)
        elif config.model_name.lower() == "wideresnet101":
            self.resnet = wide_resnet101_2(num_classes=config.hidden_size, pretrained=True)
        else:
            raise ValueError()

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.wandb = config.wandb

        self.last_losses = None
        self.last_seq_out = None
        self.last_logits = None
        self.last_labels = None
        self.last_bert_out = None

        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def verify_noise_detection(self, noise_mask, step=-1):
        if noise_mask is not None and noise_mask.sum().item() != 0:
            noise_mask = noise_mask.view(-1)

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

    def forward(self, batch, labels=None, **kwargs):
        sequence_output = self.resnet(batch)
        if torch.isnan(sequence_output).any():
            raise ValueError("NaNs in sequence output")

        batch_size, feat_dim = sequence_output.shape

        logits_network = F.softmax(self.classifier(sequence_output.view(batch_size, 1, -1)), dim=-1)
        logits = logits_network.squeeze()

        if labels is not None:
            loss_fct = nn.NLLLoss(reduction="none")

            values = torch.zeros(labels.shape[0], self.num_labels, dtype=torch.long).cuda()
            values.scatter_(1, labels.view(-1, 1),
                            torch.ones(labels.shape[0], dtype=torch.long).cuda().view(-1, 1))

            last_sequence_outputs = sequence_output.detach().view(-1, feat_dim).clone()

            loss_expanded = loss_fct(torch.log(logits), labels)

            del self.last_labels, self.last_bert_out, self.last_logits, self.last_seq_out

            self.last_seq_out = last_sequence_outputs.view(batch_size, 1, -1)
            self.last_labels = labels.detach().clone()
            self.last_logits = logits.detach().clone()
            self.last_bert_out = logits_network.detach().clone()

            loss = loss_expanded.mean()

            with torch.no_grad():
                self.last_losses = loss_expanded.detach().view(-1).clone()

                del values, loss_expanded, sequence_output

            del last_sequence_outputs, labels
            return loss, logits.view(batch_size, 1, -1)
        else:
            return logits.view(batch_size, 1, -1), sequence_output
