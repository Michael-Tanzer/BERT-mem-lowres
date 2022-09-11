from __future__ import absolute_import, division, print_function

import wandb

from consts import get_data_dirs_cardinal, get_processors, get_memories, get_reporters

from consts import get_base_parameters_trainer, get_models, get_configs, get_tokenizers

try:
    from model_utils import convert_examples_to_features, convert_examples_to_features_nli, convert_examples_to_features_ir
    from wandber import Wandber
except ImportError:
    from .model_utils import convert_examples_to_features, convert_examples_to_features_nli, convert_examples_to_features_ir
    from .wandber import Wandber

import logging
import random

import numpy as np
import torch
from pytorch_transformers import (AdamW, WarmupLinearSchedule)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self,
                 wandb=False,
                 data_dir=None,
                 model_name="roberta-base",
                 task_name="ner",
                 dataset_name="fce",
                 output_dir="./out/",
                 cache_dir="",
                 max_seq_length=128,
                 do_train=True,
                 do_eval=True,
                 eval_on="dev",  # or "test"
                 do_lower_case=False,
                 train_batch_size=32,
                 eval_batch_size=32,
                 learning_rate=5e-5,
                 num_train_epochs=5.0,
                 warmup_proportion=0.1,
                 weight_decay=0.01,
                 adam_epsilon=1e-8,
                 max_grad_norm=1.0,
                 no_cuda=False,
                 local_rank=-1,  # local_rank for distributed training on gpus
                 seed=42,
                 gradient_accumulation_steps=1,
                 # Number of updates steps to accumulate before performing a backward/update pass.
                 fp16=False,  # Whether to use 16-bit float precision instead of 32-bit
                 fp16_opt_level="O1",
                 # For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html
                 loss_scale=0,
                 # Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True. 0 (default value): dynamic loss scaling. Positive power of 2: static loss scaling value.
                 server_ip="",
                 server_port="",
                 print_every=50,
                 n_gpu=None,
                 noise_addition=0.0
                 ):

        self.data_dir = data_dir
        self.model_name = model_name
        self.task_name = task_name.lower()
        self.dataset_name = dataset_name.lower()
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length if task_name != "ir" else 1
        self.do_train = do_train
        self.do_eval = do_eval
        self.eval_on = eval_on
        self.do_lower_case = do_lower_case
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.no_cuda = no_cuda
        self.local_rank = local_rank
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.loss_scale = loss_scale
        self.server_ip = server_ip
        self.server_port = server_port
        self.print_every = print_every
        self.n_gpu = n_gpu
        self.noise_addition = noise_addition
        self.architecture_name = self.model_name.split("-")[0].split("/")[-1].upper()

        self.wandber = Wandber(wandb)

        self.eval_dataloader = None
        self.test_dataloader = None
        self.train_dataloader = None

        self.train_previous_labels = None
        self.train_forgetting_events = None
        self.train_learning_events = None
        self.train_first_learning_event = None

        self.oneshot_mask = None

        self.EMBEDDING_DIM = 1000 if self.task_name == "ir" else 768

        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.server_ip, self.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        processors = get_processors()
        data_dirs_cardinal = get_data_dirs_cardinal()
        models = get_models()
        reporters = get_reporters()
        tokenizers = get_tokenizers()
        configs = get_configs()

        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            if self.n_gpu is None:
                self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.device, self.n_gpu, bool(self.local_rank != -1), self.fp16))

        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.gradient_accumulation_steps))

        self.train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if not self.do_train and not self.do_eval:
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")

        if self.dataset_name not in processors:
            raise ValueError("Dataset not found: %s" % (self.dataset_name))

        if self.dataset_name not in data_dirs_cardinal:
            raise ValueError("Task not found: %s" % (self.task_name))

        if self.architecture_name not in models:
            raise ValueError("Model not found: %s" % (self.architecture_name))

        if self.task_name not in reporters:
            raise ValueError("Task not found: %s" % (self.task_name))

        if self.architecture_name not in tokenizers:
            raise ValueError("Tokenizer not found: %s" % (self.architecture_name))

        if self.architecture_name not in configs:
            raise ValueError("Config not found: %s" % (self.architecture_name))

        self.processor = processors[self.dataset_name]()
        self.reporter = reporters[self.task_name]
        self.model_class = models[self.architecture_name]
        self.tokenizer_class = tokenizers[self.architecture_name]
        self.config_class = configs[self.architecture_name]

        if self.data_dir is None:
            self.data_dir = data_dirs_cardinal[self.dataset_name]

        self.label_list = self.processor.get_labels()
        if self.task_name == "ir" or self.task_name == "nli":
            self.label_map = {i: label for i, label in enumerate(self.label_list, 0)}
            num_labels = len(self.label_list)
        else:
            self.label_map = {i: label for i, label in enumerate(self.label_list, 1)}
            self.label_map[0] = '[PAD]'
            num_labels = len(self.label_list) + 1

        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name, do_lower_case=self.do_lower_case)

        self.train_examples = None
        self.eval_examples = None
        self.test_examples = None
        self.num_train_optimization_steps = 0
        if self.do_train:
            self.train_examples = self.processor.get_train_examples(self.data_dir)
            self.train_forgetting_events = torch.zeros(len(self.train_examples) * self.max_seq_length, dtype=torch.long)
            self.train_learning_events = torch.zeros(len(self.train_examples) * self.max_seq_length, dtype=torch.long)
            self.train_first_learning_event = torch.zeros(len(self.train_examples) * self.max_seq_length, dtype=torch.long) - 1
            self.train_first_learning_event_misc = torch.zeros(len(self.train_examples) * self.max_seq_length, dtype=torch.long) - 1
            self.train_first_learning_event_loc = torch.zeros(len(self.train_examples) * self.max_seq_length, dtype=torch.long) - 1
            self.train_previous_labels = torch.zeros(len(self.train_examples) * self.max_seq_length, dtype=torch.bool)
            self.num_train_optimization_steps = int(
                len(self.train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_train_epochs
            if self.local_rank != -1:
                self.num_train_optimization_steps = self.num_train_optimization_steps // torch.distributed.get_world_size()

        if self.do_eval:
            self.eval_examples = self.processor.get_dev_examples(self.data_dir)
            self.test_examples = self.processor.get_test_examples(self.data_dir)

        if self.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        # Prepare model
        config = self.config_class.from_pretrained(self.model_name, num_labels=num_labels, finetuning_task=self.task_name, output_hidden_states=True)
        config.wandb = wandb

        # self.model = self.model_class(config=config)
        self.model = self.model_class.from_pretrained(self.model_name, from_tf=False, config=config)

        if self.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.cuda()

        self.setup_optimizer_and_scheduler()

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True)

        self.wandber.watch(self.model)

    def setup_optimizer_and_scheduler(self):
        param_optimizer = []

        param_optimizer += getattr(self.model, self.architecture_name.lower()).named_parameters()
        param_optimizer += list(self.model.classifier.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(self.warmup_proportion * self.num_train_optimization_steps)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=warmup_steps,
                                              t_total=self.num_train_optimization_steps)

    def train(self, start_epoch=0):
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        best_val = 0

        if self.do_train:
            self.model.train()
            train_dataloader = self.get_dataloader(train=True,
                                                   force_recompute=False,
                                                   label_noise_addition=self.noise_addition)
            total = train_dataloader.dataset.tensors[0].shape[0]

            for epoch in trange(start_epoch, int(self.num_train_epochs) + start_epoch, desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                y_true_loc_all = []
                y_true_misc_all = []
                y_pred_loc_all = []
                y_pred_misc_all = []
                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Iteration {epoch}")):
                    batch = tuple(t.cuda() for t in batch)
                    input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, noise_mask, selected_idxs = batch
                    selected_idxs = selected_idxs.cpu()

                    loss, logits = self.model(input_ids, token_type_ids=segment_ids,
                                                                     attention_mask=input_mask,
                                                                     labels=label_ids, valid_ids=valid_ids,
                                                                     attention_mask_label=l_mask,
                                                                     examples_indexes=selected_idxs,
                                                                     task="train",
                                                                     step=total * epoch + step * self.train_batch_size)

                    if self.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps

                    if self.fp16:
                        try:
                            from apex import amp
                        except ImportError:
                            raise ImportError(
                                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    tr_loss += loss.item()
                    # nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.model.verify_noise_detection(noise_mask, step=total * epoch + step * self.train_batch_size)

                    train_new_labels = (logits.argmax(2).squeeze() == label_ids).view(-1).cpu()
                    selected_idxs = selected_idxs.view(-1)
                    current_batch_selector = torch.zeros_like(self.train_learning_events)
                    current_batch_selector[selected_idxs] = 1
                    current_batch_selector = current_batch_selector.bool()
                    correctly_classified_selector = torch.zeros_like(current_batch_selector).bool()
                    correctly_classified_selector[selected_idxs] = train_new_labels
                    incorrectly_classified_selector = torch.zeros_like(current_batch_selector).bool()
                    incorrectly_classified_selector[selected_idxs] = ~train_new_labels
                    self.train_forgetting_events[current_batch_selector & self.train_previous_labels & incorrectly_classified_selector] += 1
                    self.train_learning_events[current_batch_selector & (~self.train_previous_labels) & correctly_classified_selector] += 1
                    self.train_previous_labels[selected_idxs] = train_new_labels

                    label_ids_selector = torch.zeros_like(current_batch_selector).long()
                    label_ids_selector[selected_idxs] = label_ids.view(-1).cpu()
                    self.train_first_learning_event_misc[correctly_classified_selector & current_batch_selector & (self.train_first_learning_event_misc==-1).bool() & ((label_ids_selector == 2) | (label_ids_selector == 3)) ] = epoch
                    self.train_first_learning_event_loc[correctly_classified_selector & current_batch_selector & (self.train_first_learning_event_loc==-1).bool() & ((label_ids_selector == 8) | (label_ids_selector == 9)) ] = epoch

                    self.train_first_learning_event[correctly_classified_selector & current_batch_selector & (self.train_first_learning_event==-1).bool()] = epoch

                    y_true, y_pred = self.get_labels(logits.detach(), label_ids.detach())
                    y_true_loc = [[word if "LOC" in word else "O" for word in sentence] for sentence in y_true]
                    y_true_misc = [[word if "MISC" in word else "O" for word in sentence] for sentence in y_true]
                    y_pred_loc = [[word if "LOC" in word else "O" for word in sentence] for sentence in y_pred]
                    y_pred_misc = [[word if "MISC" in word else "O" for word in sentence] for sentence in y_pred]

                    y_true_loc_all.extend(y_true_loc)
                    y_true_misc_all.extend(y_true_misc)
                    y_pred_loc_all.extend(y_pred_loc)
                    y_pred_misc_all.extend(y_pred_misc)

                    if self.wandber.on:
                        with torch.no_grad():
                            lm = {v:k for k,v in self.label_map.items()}
                            entropy = lambda x: -(x * x.log()).sum().item()
                            correct_logits = logits.max(2).values[(logits.argmax(2) == label_ids) & (label_ids != lm["[PAD]"]) & (label_ids != lm["[SEP]"]) & (label_ids != lm["[CLS]"])]
                            incorrect_logits = logits.max(2).values[(logits.argmax(2) != label_ids) & (label_ids != lm["[PAD]"]) & (label_ids != lm["[SEP]"]) & (label_ids != lm["[CLS]"])]
                            valid_logits = logits.max(2).values[(label_ids != lm["[PAD]"]) & (label_ids != lm["[SEP]"]) & (label_ids != lm["[CLS]"])]

                            summed_weights = 0
                            summed_gradients = 0
                            for param in list(getattr(self.model, self.architecture_name.lower()).parameters()) + list(self.model.classifier.parameters()):
                                summed_weights += torch.norm(param).item()
                                if param.grad is not None:
                                    summed_gradients += torch.norm(param.grad).item()

                            wandb.log({
                                        "entropy/entropy-correct": entropy(correct_logits),
                                        "entropy/entropy-incorrect": entropy(incorrect_logits),
                                        "entropy/entropy-predicted": entropy(valid_logits),
                                        "entropy/logits-correct": correct_logits.mean(),
                                        "entropy/logits-incorrect": incorrect_logits.mean(),
                                        "entropy/logits-predicted": valid_logits.mean(),
                                        "entropy/summed-weight": summed_weights,
                                        "entropy/summed-gradients": summed_gradients
                                       },
                                      step=total * epoch + step * self.train_batch_size)

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1

                    if ((global_step + 1) % self.print_every == 0 or global_step == 1) and self.do_eval:
                        with torch.no_grad():
                            y_true, y_pred = self.get_labels(logits.detach(), label_ids.detach())
                            metrics = self.reporter(y_true, y_pred)
                            keys = ["accuracy", "precision", "recall", "f05score", "f1score"]
                            self.print_metrics(epoch, step, {k: metrics[k] * 100 for k in keys}, train=True)

                            metrics["loss"] = loss.item()
                            self.wandber.log_training_step({k: v for k, v in metrics.items() if k != "report"}, step=total * epoch + step * self.train_batch_size)
                            if self.wandber.on and metrics["report"] is not None:
                                wandb.log({
                                        f"training-report": wandb.Table(
                                            columns=[""] + metrics["report"].split("\n")[0].split(),
                                            data=[t.split() for i, t in enumerate(metrics["report"].split("\n")) if
                                                  (0 < i < len(metrics["report"].split("\n")) - 3) and (len(t) > 0)])
                                }, step=total * epoch + step * self.train_batch_size)

                            y_true_loc = [[word if "LOC" in word else "O" for word in sentence] for sentence in y_true]
                            y_true_misc = [[word if "MISC" in word else "O" for word in sentence] for sentence in y_true]
                            print(sum([sum([1 if "MISC" in word else 0 for word in sentence]) for sentence in y_true]))
                            y_pred_loc = [[word if "LOC" in word else "O" for word in sentence] for sentence in y_pred]
                            y_pred_misc = [[word if "MISC" in word else "O" for word in sentence] for sentence in y_pred]

                            metrics_loc = self.reporter(y_true_loc, y_pred_loc, skipreport=True)
                            metrics_misc = self.reporter(y_true_misc, y_pred_misc, skipreport=True)

                            self.wandber.log_training_step({"locmetricstrain_" + k: v for k, v in metrics_loc.items() if k != "report"},
                                                           step=total * epoch + step * self.train_batch_size)
                            self.wandber.log_training_step({"miscmetricstrain_" + k: v for k, v in metrics_misc.items() if k != "report"},
                                                           step=total * epoch + step * self.train_batch_size)

                    if ((global_step+1) % (2 * self.print_every) == 0 or global_step == 1) and self.do_eval:
                        eval_metrics = self.eval(load_model=False, intermediate=True, epoch=epoch,
                                                 step=total * epoch + step * self.train_batch_size)
                        if eval_metrics["f1score"] >= best_val and global_step != 1:
                            best_val = eval_metrics["f1score"]
                            self.eval_on = "test"
                            self.eval(load_model=False, intermediate=False,
                                      step=total * epoch + step * self.train_batch_size)
                            self.eval_on = "dev"
                        self.model.train()

                    print(f"before del - {torch.cuda.memory_allocated()}")
                    del input_ids, input_mask, segment_ids, label_ids, valid_ids, \
                        l_mask, noise_mask, logits, loss

                    torch.cuda.empty_cache()


                metrics_loc_all = self.reporter(y_true_loc_all, y_pred_loc_all, skipreport=True)
                metrics_misc_all = self.reporter(y_true_misc_all, y_pred_misc_all, skipreport=True)

                self.wandber.log_training_step(
                    {"locmetricstrain_all_" + k: v for k, v in metrics_loc_all.items() if k != "report"},
                    step=total * epoch + step * self.train_batch_size)
                self.wandber.log_training_step(
                    {"miscmetricstrain_all_" + k: v for k, v in metrics_misc_all.items() if k != "report"},
                    step=total * epoch + step * self.train_batch_size)


                eval_metrics = self.eval(load_model=False, intermediate=True, epoch=epoch,
                                         step=total * epoch + step * self.train_batch_size)
                if eval_metrics["f1score"] >= best_val and global_step != 1:
                    best_val = eval_metrics["f1score"]
                    self.eval_on = "test"
                    self.eval(load_model=False, intermediate=False,
                              step=total * epoch + step * self.train_batch_size)
                    self.eval_on = "dev"
                self.model.train()

            k = 20
            highest_idx = self.train_forgetting_events.argsort(descending=True)[:k]
            examples = [self.train_examples[id // self.max_seq_length] for id in highest_idx]

            torch.save(self.train_forgetting_events, "forgetting_events.pt")
            torch.save(self.train_learning_events, "learning_events.pt")
            torch.save(self.train_first_learning_event, "first_learning_events.pt")

            if wandb.run is not None:
                if self.task_name == "ir":
                    wandb.log({"forgetting/most-forgotten-examples": [wandb.Image(ex.image, caption=f"{self.train_forgetting_events[highest_idx[i]]} forgetting events") for i, ex in enumerate(examples)]})
                elif self.task_name == "ged" or self.task_name == "ner":
                    wandb.log({"forgetting/most-forgotten-examples": wandb.Table(data=[(self.train_forgetting_events[highest_idx[i]], ex.text_a, (highest_idx[i] % self.max_seq_length)) for i, ex in enumerate(examples)], columns=["forgetting events", "sentence", "position"])})
                elif self.task_name == "nli":
                    wandb.log({"forgetting/most-forgotten-examples": [f"{ex.text_a} | {ex.text_b}" for ex in examples]})

                wandb.log({"forgetting/forgetting-events": wandb.Histogram(self.train_forgetting_events)})
                # wandb.log({"forgetting-events": self.train_forgetting_events})
                wandb.log({"forgetting/learning-events": wandb.Histogram(self.train_learning_events)})
                wandb.log({"forgetting/first-learning-event": wandb.Histogram(self.train_first_learning_event)})

                wandb.log({"forgetting/first-learning-event-misc": wandb.Histogram(self.train_first_learning_event_misc)})
                wandb.log({"forgetting/first-learning-event-loc": wandb.Histogram(self.train_first_learning_event_loc)})

                if self.noise_addition > 0:
                    noise_mask = self.train_dataloader.dataset.tensors[-2].view(-1).bool()
                    forgetting_noise = self.train_forgetting_events[noise_mask]
                    forgetting_not_noise = self.train_forgetting_events[~noise_mask]
                    wandb.run.summary["forgetting/avg-noisy"] = forgetting_noise.float().mean()
                    wandb.run.summary["forgetting/std-noisy"] = forgetting_noise.float().std()
                    wandb.run.summary["forgetting/avg-not-noisy"] = forgetting_not_noise.float().mean()
                    wandb.run.summary["forgetting/std-not-noisy"] = forgetting_not_noise.float().std()


                wandb.run.summary["forgetting/number-unforgettable"] = (self.train_forgetting_events == 0).long().sum()
                wandb.run.summary["forgetting/number-forgettable"] = (self.train_forgetting_events != 0).long().sum()
                wandb.run.summary["forgetting/number-learned"] = (self.train_first_learning_event != -1).long().sum()
                wandb.save("forgetting_events.pt")
                wandb.save("learning_events.pt")
                wandb.save("first_learning_events.pt")

            # # Save a trained model and the associated configuration
            # model_to_save = self.model.module if hasattr(self.model,
            #                                              'module') else self.model  # Only save the model it-self
            # model_to_save.save_pretrained(self.output_dir)
            # self.tokenizer.save_pretrained(self.output_dir)
            # model_config = {"bert_model": self.bert_model, "do_lower": self.do_lower_case,
            #                 "max_seq_length": self.max_seq_length, "num_labels": len(self.label_list) + 1,
            #                 "label_map": self.label_map}
            # json.dump(model_config, open(os.path.join(self.output_dir, "model_config.json"), "w"))
            # # Load a trained model and config that you have fine-tuned

    def eval(self, load_model=True, intermediate=False, epoch=0, step=0):
        if self.do_eval and (self.local_rank == -1 or torch.distributed.get_rank() == 0):
            if load_model:
                # Load a trained model and vocabulary that you have fine-tuned
                self.model = self.model_class.from_pretrained(self.output_dir)
                self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name, do_lower_case=self.do_lower_case)
                self.model.cuda()

                self.model.start_mem = 0

            eval_dataloader = self.get_dataloader(train=False, force_recompute=False)
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            y_pred_bm = []
            y_pred_mem = []

            with torch.no_grad():
                for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, noise_mask, selected_idxs in eval_dataloader:
                    input_ids = input_ids.cuda()
                    input_mask = input_mask.cuda()
                    segment_ids = segment_ids.cuda()
                    valid_ids = valid_ids.cuda()
                    label_ids = label_ids.cuda()
                    l_mask = l_mask.cuda()
                    noise_mask = noise_mask.cuda()
                    selected_idxs = selected_idxs.cuda()

                    logits, logits_bm = self.model(input_ids, token_type_ids=segment_ids,
                                                                     attention_mask=input_mask,
                                                                     valid_ids=valid_ids,
                                                                     attention_mask_label=l_mask,
                                                                     examples_indexes=selected_idxs,
                                                                     task=self.eval_on,
                                                                     step=step
                                                                  )

                    logits = logits.cpu().detach()
                    logits_bm = logits_bm.cpu().detach()
                    label_ids = label_ids.cpu().detach()


                    new_true, new_pred = self.get_labels(logits, label_ids)
                    y_true.extend(new_true)
                    y_pred.extend(new_pred)

                    del input_ids, input_mask, segment_ids, label_ids, valid_ids, \
                        l_mask, logits, logits_bm

                    torch.cuda.empty_cache()

            metrics = self.reporter(y_true, y_pred, digits=4)

            if intermediate:
                keys = ["accuracy", "precision", "recall", "f05score", "f1score"]
                self.print_metrics(epoch, step, {k: metrics[k] * 100 for k in keys}, train=False)
                self.wandber.log_validation_step({k:v for k,v in metrics.items() if k != "report"}, step)
                if self.wandber.on and metrics["report"] is not None:
                    wandb.log({
                            f"{'validation' if self.eval_on == 'dev' else 'test'}-report": wandb.Table(columns=[""] + metrics["report"].split("\n")[0].split(),
                                                                   data=[t.split() for i, t in enumerate(metrics["report"].split("\n")) if
                                                                         (0 < i < len(metrics["report"].split("\n")) - 3) and (len(t) > 0)])
                    }, step=step)

                y_true_loc = [[word if "LOC" in word else "O" for word in sentence] for sentence in y_true]
                y_true_misc = [[word if "MISC" in word else "O" for word in sentence] for sentence in y_true]
                y_pred_loc = [[word if "LOC" in word else "O" for word in sentence] for sentence in y_pred]
                y_pred_misc = [[word if "MISC" in word else "O" for word in sentence] for sentence in y_pred]

                metrics_loc = self.reporter(y_true_loc, y_pred_loc, skipreport=True)
                metrics_misc = self.reporter(y_true_misc, y_pred_misc, skipreport=True)

                self.wandber.log_validation_step(
                    {"locmetricstrain_" + k: v for k, v in metrics_loc.items() if k != "report"},
                    step=step)
                self.wandber.log_validation_step(
                    {"miscmetricstrain_" + k: v for k, v in metrics_misc.items() if k != "report"},
                    step=step)
            else:
                report = metrics["report"]
                logger.info("\n%s", report)

                metrics.pop("report")
                self.wandber.log_summary_metrics(metrics, name=self.eval_on)

            return metrics

    def get_labels(self, logits, label_ids):
        y_true = []
        y_pred = []
        if self.task_name == "ir":
            label_ids = label_ids.view(-1,1)
            y_pred = torch.argmax(logits[:, 0, :], dim=1)
            y_true = label_ids[:, 0]
            y_pred = [[self.label_map[y.item()] for y in y_pred]]
            y_true = [[self.label_map[y.item()] for y in y_true]]
        elif self.task_name == "nli":
            y_pred = torch.argmax(logits[:,0,:], dim=1)
            y_true = label_ids[:,0]
            y_pred = [[self.label_map[y.item()] for y in y_pred]]
            y_true = [[self.label_map[y.item()] for y in y_true]]
        else:
            logits = torch.argmax(logits, dim=2)

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(self.label_map) - 1:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(self.label_map[label_ids[i][j].item()])
                        temp_2.append(self.label_map[logits[i][j].item()])

        return y_true, y_pred

    def get_dataloader(self, force_recompute=False, train=True, label_noise_addition=0.0):
        if (train and self.train_dataloader is None) or \
            (not train and self.eval_on == "dev" and self.eval_dataloader is None) or \
            (not train and self.eval_on == "test" and self.test_dataloader is None) or \
                force_recompute:

            batch_size = self.train_batch_size if train else self.eval_batch_size

            if train:
                examples = self.train_examples
            else:
                if self.eval_on == "dev":
                    examples = self.eval_examples
                elif self.eval_on == "test":
                    examples = self.test_examples
                else:
                    raise ValueError("eval on dev or test set only")

            if self.task_name == "nli":
                features = convert_examples_to_features_nli(examples, self.label_list, self.max_seq_length, self.tokenizer)
            elif self.task_name == "ir":
                features = convert_examples_to_features_ir(examples, self.label_list)
            else:
                features = convert_examples_to_features(examples, self.label_list, self.max_seq_length, self.tokenizer)

            # logger.info(f"***** Running {'training' if train else 'evaluation'} *****")
            # logger.info(f"  Num examples = {len(examples)}")
            # logger.info(f"  Batch size = {batch_size}")
            # if train: logger.info("  Num steps = %d", self.num_train_optimization_steps)
            if self.task_name == "ir":
                all_input_ids = torch.cat([f.input_ids.unsqueeze(0) for f in features], dim=0)
            else:
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

            noise_selector = torch.rand_like(all_label_ids, dtype=torch.float) < label_noise_addition

            if 0 < label_noise_addition <= 1:
                noise = torch.zeros_like(all_label_ids, dtype=torch.float)
                noise.uniform_(1, len(self.label_list) if self.task_name == "ir" else len(self.label_list)-1)
                noise = noise.long()
                if self.task_name != "ir":
                    inverse_map = {v: k for k, v in self.label_map.items()}
                    noise_selector[all_label_ids == inverse_map['[CLS]']] = False
                    noise_selector[all_label_ids == inverse_map['[PAD]']] = False
                    noise_selector[all_label_ids == inverse_map['[SEP]']] = False
                noise_selector[noise == all_label_ids] = False
                all_label_ids[noise_selector] = noise[noise_selector]

                logger.info(f"Was supposed to add {label_noise_addition*100}% noise - Adding {noise_selector.float().mean()}% noise, corresponding to {noise_selector.int().sum()} wrong labels")

                torch.save(noise_selector.long().view(-1), f'noise_mask_{self.model_name}_{self.dataset_name}_{self.noise_addition}.pt')

            if self.task_name == "ir":
                idxs = torch.tensor([int(i) for i in range(all_input_ids.shape[0])])
            else:
                idxs = torch.tensor([int(i) for i in range(all_input_ids.view(-1).shape[0])]).view_as(all_input_ids)

            all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
            all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

            # -------- Noise correction using noise detection ---------------
            # if train and self.task_name != "ir" and 0 < label_noise_addition <= 1:
            #     inverse_map = {v: k for k, v in self.label_map.items()}
            #     believed_noisy = torch.load('../experiments/train_noise_pred.pt')
            #     believed_noisy = believed_noisy.view_as(all_lmask_ids)
            #     believed_noisy[all_input_ids == inverse_map['[CLS]']] = 0
            #     believed_noisy[all_input_ids == inverse_map['[PAD]']] = 0
            #     believed_noisy[all_input_ids == inverse_map['[SEP]']] = 0
            #     all_lmask_ids[believed_noisy == 1] = 0
            #     print((believed_noisy.view(-1) == noise_selector.view(-1)).float().mean())
            # ----------------------------------------------------------------

            # ------- mask ignore part of dataset based on forgetting --------
            # if train:
            #     f = torch.load("forgetting_events_ref.pt")  # base
            #     mask = (f == 0).view(-1, self.max_seq_length).squeeze()  # base
            #
            #     p = (mask.float().sum() / mask.view(-1).shape[0]).item()  # random
            #     # p = 0.32729291915893555
            #     random_1p_mask = torch.rand(mask.shape) < p  # random
            #     all_lmask_ids[random_1p_mask] = 0  #  random
            #
                # random_mask = torch.rand(mask.shape) >= 0.  # forgetting
                # all_lmask_ids[mask & random_mask] = 0  # forgetting
            #
            #     f2 = torch.load("models/confidences_conll03_ref.pt")
            #     n = int(f2.shape[0] * (1-p))
            #     print(n)
            #     cutoff = f2.sort(descending=True).values[n].item()
            #     confidence_mask = (f2 < cutoff).view(-1, self.max_seq_length).squeeze()
            #     print((confidence_mask.long().sum() / confidence_mask.shape[0]).item())
            #     all_lmask_ids[confidence_mask] = 0

                # f3 = torch.load('learning_events_ref_noisy.pt')  # unlearnable
                # mask = (f3 != 0).view(-1, self.max_seq_length).squeeze()  # unlearnable
                # all_lmask_ids[mask] = 0  # unlearnable
            # -----------------------------------------------------------------

            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                 all_lmask_ids, noise_selector.long(), idxs.long())

            if self.local_rank == -1 and train:
                sampler = RandomSampler(data)
            elif train:
                sampler = DistributedSampler(data)
            else:
                sampler = SequentialSampler(data)

            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

            if train:
                self.train_dataloader = dataloader
            else:
                if self.eval_on == "dev":
                    self.eval_dataloader = dataloader
                elif self.eval_on == "test":
                    self.test_dataloader = dataloader

        if train:
            return self.train_dataloader
        else:
            if self.eval_on == "dev":
                return self.eval_dataloader
            elif self.eval_on == "test":
                return self.test_dataloader

    @staticmethod
    def print_metrics(epoch, i, stats, train=True):
        start = f"\n[{epoch}, {i + 1: 3}]"
        name = f"{'Trn' if train else 'Val'} stats:"
        stats = [f"{k}: {v:.3f}" if len(k) != 2 else f"{k}: {v}" for k, v in stats.items()]
        to_log = [start, name] + stats
        to_log = "".join([s.rjust(20) for s in to_log])
        logger.info(to_log)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Model trainer')
    for k, v in get_base_parameters_trainer().items():
        if type(v) == bool:
            parser.add_argument(f"--{k.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)

    parser.add_argument("--paper", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    hyperparameter_defaults = vars(args)

    if hyperparameter_defaults["wandb"]:
        run = wandb.init(name="", config=hyperparameter_defaults, project="bert-memorisation-and-pitfalls-paper2", tags=[
                    hyperparameter_defaults["model_name"].split("-")[0].upper(),
                    "PAPER3",
                    "LOWRESLOGALL3"
                ])

        config = wandb.config
        wandb.save("*.py")
        wandb.save("./utils_/*.py", base_path="..")
        print("using wandb")
    else:
        config = hyperparameter_defaults
        print("not using wandb")

    config = dict(config)
    if "paper" in config:
        config.pop("paper")

    t = Trainer(**config)
    t.train()

    # t.train(oneshot_split=-2)

    # t.train(oneshot_split=1)
    # t.num_train_epochs = 1.0
    # t.num_train_optimization_steps = int(
    #     len(t.train_examples) / t.train_batch_size / t.gradient_accumulation_steps) * t.num_train_epochs
    # t.setup_optimizer_and_scheduler()
    # t.train_dataloader = t.get_dataloader(force_recompute=True, oneshot_split=2)
    # t.train(oneshot_split=2, start_epoch=4)

    t.eval_on = "test"
    t.eval(load_model=False, intermediate=False)


    # t.eval_on = "test"
    # t.eval(load_model=False, intermediate=False)
    # t.add_noise_and_detect()
    # t.eval_knn()

    # ------------

