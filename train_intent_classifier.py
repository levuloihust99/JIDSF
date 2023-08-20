import os
import json
import time
import torch
import logging
import argparse
from torch import nn
from datetime import datetime
from tqdm import tqdm

from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

from typing import Text, List, Dict, Any

from transformers import (
    BertTokenizer, PhobertTokenizer,
    BertModel, RobertaModel,
    BertPreTrainedModel, RobertaPreTrainedModel,
    RobertaForSequenceClassification, BertForSequenceClassification
)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from utils.logging_utils import add_color_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)

with open("configs/intent_cls_config.json", "r") as reader:
    cfg = json.load(reader)
cfg = argparse.Namespace(**cfg)
data_path = "data/train.jsonl"
train_indices_path = "data/split/train_indices.json"
dev_indices_path = "data/split/dev_indices.json"

tokenizer_path = "vinai/phobert-base"
tokenizer_type = "phobert"
model_path = "vinai/phobert-base"
model_type = "phobert"

max_seq_length = 256
train_batch_size = 16
eval_batch_size = 16

weight_decay = 0.1
learning_rate = 5e-5
adam_epsilon = 1e-8
num_train_epochs = 10
gradient_accumulation_steps = 1
warmup_proportion = 0.1
max_steps = -1
warmup_steps = 0
seed = 12345
gpu_id = 0
checkpoint_dir = "checkpoints/intent_cls"


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_data():
    all_data = []
    with open(cfg.data_path, "r") as reader:
        for line in reader:
            all_data.append(json.loads(line.strip()))

    with open(cfg.train_indices_path, "r") as reader:
        train_indices = json.load(reader)
    with open(cfg.dev_indices_path, "r") as reader:
        dev_indices = json.load(reader)

    all_intents = set()
    train_data = []
    for idx in train_indices:
        train_data.append(all_data[idx])
        all_intents.add(all_data[idx]["intent"])
    dev_data = []
    for idx in dev_indices:
        all_intents.add(all_data[idx]["intent"])
        dev_data.append(all_data[idx])

    all_intents = list(all_intents)
    tag2int = {intent: idx for idx, intent in enumerate(all_intents)}
    int2tag = {v: k for k, v in tag2int.items()}

    return train_data, dev_data, tag2int, int2tag


def load_tokenizer():
    if cfg.tokenizer_type == "phobert":
        return PhobertTokenizer.from_pretrained(cfg.tokenizer_path)
    elif cfg.tokenizer_type == "bert":
        return BertTokenizer.from_pretrained(cfg.tokenizer_path)
    else:
        raise Exception("Tokenizer type '{}' is not supported.".format(cfg.tokenizer_type))


def load_model(num_labels):
    if cfg.model_type == "phobert":
        return RobertaForSequenceClassification.from_pretrained(
            cfg.model_path, num_labels=num_labels, problem_type="single_label_classification")
    elif cfg.model_type == "bert":
        return BertForSequenceClassification.from_pretrained(
            cfg.model_path, num_labels=num_labels, problem_type="single_label_classification")
    else:
        raise Exception("Model type '{}' is not supported.".format(cfg.model_type))


def get_collate_fn(tokenizer, tag2int):
    def collate_fn(items: List[Dict[Text, Text]]):
        texts = []
        intents = []
        for item in items:
            texts.append(item["sentence"])
            intents.append(item["intent"])

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_length,
            return_tensors="pt"
        )

        labels = torch.tensor([tag2int[intent] for intent in intents], dtype=torch.int64)
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels
        }
    return collate_fn


def create_dataloader(
    data: List[Dict[Text, Any]],
    tokenizer, tag2int,
    batch_size: int,
    training: bool = True,
):
    dataset = Dataset.from_list(data)
    collate_fn = get_collate_fn(tokenizer, tag2int)
    data_loader = DataLoader(dataset, shuffle=training, batch_size=batch_size, collate_fn=collate_fn)
    return data_loader


def create_optimizer_and_scheduler(model, total_steps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    num_warmup_steps_by_ratio = int(total_steps * cfg.warmup_proportion)
    num_warmup_steps_absolute = cfg.warmup_steps
    if num_warmup_steps_absolute == 0 or num_warmup_steps_by_ratio == 0:
        num_warmup_steps = max(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    else:
        num_warmup_steps = min(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler


class IntentClassifierTrainer:
    def __init__(
        self,
        tokenizer,
        model,
        train_data_loader,
        dev_data_loader,
        scheduler,
        optimizer,
        tag2int,
        int2tag
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.tag2int = tag2int
        self.int2tag = int2tag

        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}, {}'.format(torch.cuda.get_device_name(gpu_id), torch.cuda.get_device_capability(gpu_id)))
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     logger.info("MPS backend is available, using MPS.")
        else:
            logger.info('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
    

    def train(self):
        self.best_result = 0
        self.best_report = None

        self.model.train()
        for epoch in range(0, num_train_epochs):
            logger.info(' Epoch {:} / {:}'.format(epoch + 1, num_train_epochs))

            t0 = time.time()
            total_loss = 0
            self.model.train()
            progress_bar = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader))

            for step, batch in enumerate(self.train_data_loader):
                batch_input_ids = batch['input_ids'].to(self.device)
                batch_attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)

                batch_outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    return_dict=True
                )

                batch_loss = batch_outputs.loss
                total_loss += batch_loss.item()
            
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": float(batch_loss)})
            
            avg_train_loss = total_loss / len(self.train_dataloader)
            logger.info("Average training loss: {0:.5f}".format(avg_train_loss))
            logger.info("Training epoch took: {:}".format(format_time(time.time() - t0)))
            self.eval()
    
    def eval(self):
        logger.info("Running Validation...")
        self.model.eval()
        t0 = time.time()

        eval_tracker = {
            i: {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "Support": 0
            } for tag, i in self.tag2int.items()
        }
        for batch in self.dev_data_loader:
            batch = {k : v.to(self.device) for k, v in batch.items()}
            batch_input_ids = batch["input_ids"]
            batch_attention_mask = batch["attention_mask"]
            batch_labels = batch["labels"]

            with torch.no_grad():
                batch_outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_dict=True
                )
            
            batch_logits = batch_outputs.logits
            batch_preds = torch.argmax(batch_logits, dim=-1)

            for pred, label in zip(batch_labels, batch_preds):
                pred = pred.item()
                label = label.item()
                if pred != label:
                    eval_tracker[pred]["FP"] += 1
                    eval_tracker[label]["FN"] += 1
                else:
                    eval_tracker[pred]["TP"] += 1
                eval_tracker[label]["Support"] += 1
            
        eval_results = {}
        for tag_id, metrics in eval_tracker.items():
            precision = metrics["TP"] / (metrics["TP"] + metrics["FP"])
            recall = metrics["TP"] / (metrics["TP"] + metrics["FN"])
            if precision == 0 or recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            eval_results[self.int2tag[tag_id]] = {
                "Precision": metrics["TP"] / (metrics["TP"] + metrics["FP"]),
                "Recall": metrics["TP"] / (metrics["TP"] + metrics["FN"]),
                "F1-score": f1
            }
        
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        for metrics in eval_results.values():
            precisions.append(metrics["Precision"])
            recalls.append(metrics["Recall"])
            f1_scores.append(metrics["F1-score"])
            supports.append(metrics["Support"])
        total = sum(supports)
        
        macro_p = sum(precisions) / len(precisions)
        macro_r = sum(recalls) / len(recalls)
        if macro_p == 0.0 or macro_r == 0.0:
            macro_f = 0.0
        else:
            macro_f = (
                2 * macro_p * macro_r / (macro_p + macro_r)
            )
        macro_eval = {
            "Precision": macro_p,
            "Recall": macro_r,
            "F1-score": macro_f,
            "Support": total
        }

        total = sum(supports)
        weights = [s / total for s in supports]
        weighted_p = 0.0
        weighted_r = 0.0
        for w, p, r, f in zip(weights, precisions, recalls, f1_scores):
            weighted_p += w * p
            weighted_r += w * r
        
        if weighted_p == 0.0 or weighted_r == 0.0:
            weighted_f = 0.0
        else:
            weighted_f = (
                2 * weighted_p * weighted_r /
                (weighted_p + weighted_r)
            )
        weighted_eval = {
            "Precision": weighted_p,
            "Recall": weighted_r,
            "F1-score": weighted_f,
            "Support": total
        }

        sum_TP = 0
        sum_FP = 0
        sum_FN = 0
        for metrics in eval_tracker.values():
            sum_TP += metrics["TP"]
            sum_FP += metrics["FP"]
            sum_FN += metrics["FN"]
        micro_p = sum_TP / (sum_TP + sum_FP)
        micro_r = sum_TP / (sum_TP + sum_FN)
        if micro_p == 0.0 or micro_r == 0.0:
            micro_f = 0.0
        else:
            micro_f = (
                2 * micro_p * micro_r / (micro_p + micro_r)
            )
        micro_eval = {
            "Precision": micro_p,
            "Recall": micro_r,
            "F1-score": micro_f,
            "Support": total
        }
        
        # format eval results
        max_tag_length = 0
        for tag in self.tag2int:
            if max_tag_length < len(tag):
                max_tag_length = len(tag)
        if len("Micro") < max_tag_length:
            max_tag_length = len("Micro")
        if len("Macro") < max_tag_length:
            max_tag_length = len("Macro")
        if len("Weighted") < max_tag_length:
            max_tag_length = len("Weighted")

        max_p_length = len("Precision")
        max_r_length = len("Recall")
        max_f_length = len("F1-score")
        max_support_length = len("Support")
        p_cols = []
        r_cols = []
        f_cols = []
        support_cols = []
        for metrics in eval_results.values():
            p = "{:10.4f}".format(metrics["Precision"])
            if max_p_length < len(p):
                max_p_length = len(p)
            r = "{:10.4f}".format(metrics["Recall"])
            if max_r_length < len(r):
                max_r_length = len(r)
            f = "{:10.4f}".format(metrics["F1-score"])
            if max_f_length < len(f):
                max_f_length = len(f)
            support = "{:10d}".format(metrics["Support"])
            if max_support_length < len(support):
                max_support_length = len(support)
            p_cols.append(p)
            r_cols.append(r)
            f_cols.append(f)
            support_cols.append(support)
        
        micro_p_str = "{:10.4f}".format(micro_eval["Precision"])
        micro_r_str = "{:10.4f}".format(micro_eval["Recall"])
        micro_f_str = "{:10.4f}".format(micro_eval["F1-score"])
        micro_support_str = str(micro_eval["Support"])
        
        macro_p_str = "{:10.4f}".format(macro_eval["Precision"])
        macro_r_str = "{:10.4f}".format(macro_eval["Recall"])
        macro_f_str = "{:10.4f}".format(macro_eval["F1-score"])
        macro_support_str = str(macro_eval["Support"])

        weighted_p_str = "{:10.4f}".format(weighted_eval["Precision"])
        weighted_r_str = "{:10.4f}".format(weighted_eval["Recall"])
        weighted_f_str = "{:10.4f}".format(weighted_eval["F1-score"])
        weighted_support_str = str(weighted_eval["Support"])

        max_p_length = max(
            len(micro_p_str),
            len(macro_p_str),
            len(weighted_p_str),
            max_p_length
        )
        max_r_length = max(
            len(micro_r_str),
            len(macro_r_str),
            len(weighted_r_str),
            max_r_length
        )
        max_f_length = max(
            len(micro_f_str),
            len(macro_f_str),
            len(weighted_f_str),
            max_f_length
        )
        max_support_length = max(
            len(micro_support_str),
            len(macro_support_str),
            len(weighted_support_str),
            max_support_length
        )
        
        tag_col_width = max_tag_length + 5
        p_col_width = max_p_length + 5
        r_col_width = max_r_length + 5
        f_col_width = max_f_length + 5
        support_col_width = max_support_length + 5
        eval_str = ""
        eval_str += " " * tag_col_width
        eval_str += "{{:>{}s}}".format(p_col_width).format("Precision")
        eval_str += "{{:>{}s}}".format(r_col_width).format("Recall")
        eval_str += "{{:>{}s}}".format(f_col_width).format("F1-score")
        eval_str += "{{:>{}s}}".format(support_col_width).format("Support")
        eval_str += "\n"
        tags = list(self.tag2int.keys())

        for tag, p, r, f, support in zip(tags, p_cols, r_cols, f_cols, support_cols):
            eval_str += "{{:{}s}}".format(tag_col_width).format(tag)
            eval_str += "{{:>{}s}}".format(p_col_width).format(p)
            eval_str += "{{:>{}s}}".format(r_col_width).format(r)
            eval_str += "{{:>{}s}}".format(f_col_width).format(f)
            eval_str += "{{:>{}s}}".format(support_col_width).format(support)
            eval_str += "\n"
        
        eval_str += "\n"
        eval_str += "{{:{}s}}".format(tag_col_width).format("Micro")
        eval_str += "{{:>{}s}}".format(p_col_width).format(micro_p_str)
        eval_str += "{{:>{}s}}".format(r_col_width).format(micro_r_str)
        eval_str += "{{:>{}s}}".format(f_col_width).format(micro_f_str)
        eval_str += "{{:>{}s}}".format(support_col_width).format(micro_support_str)

        eval_str += "\n"
        eval_str += "{{:{}s}}".format(tag_col_width).format("Macro")
        eval_str += "{{:>{}s}}".format(p_col_width).format(macro_p_str)
        eval_str += "{{:>{}s}}".format(r_col_width).format(macro_r_str)
        eval_str += "{{:>{}s}}".format(f_col_width).format(macro_f_str)
        eval_str += "{{:>{}s}}".format(support_col_width).format(macro_support_str)

        eval_str += "\n"
        eval_str += "{{:{}s}}".format(tag_col_width).format("Weighted")
        eval_str += "{{:>{}s}}".format(p_col_width).format(weighted_p_str)
        eval_str += "{{:>{}s}}".format(r_col_width).format(weighted_r_str)
        eval_str += "{{:>{}s}}".format(f_col_width).format(weighted_f_str)
        eval_str += "{{:>{}s}}".format(support_col_width).format(weighted_support_str)

        eval_str += "\n"

        logger.info("***** Eval results *****")
        logger.info("\n%s", eval_str)

        if self.best_result == 0 or self.best_result < micro_eval["F1-score"]:
            output_dir = os.path.join(checkpoint_dir,
                                    'checkpoint-{}-{}-{:.3f}'.format(self.model.__class__.__qualname__,
                                                                    learning_rate,
                                                                    micro_eval['F1-score']))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(eval_str)

            self.best_result = micro_eval['F1-score']

            print("Saving model to %s" % output_dir)

            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            with open(os.path.join(cfg.output_dir, "label_mappings.json"), "w") as writer:
                json.dump(self.tag2int, writer, indent=4, ensure_ascii=False)
            with open(os.path.join(cfg.checkpoint_dir, "training_config.json"), "w") as writer:
                json.dump(cfg.__dict__, writer, indent=4, ensure_ascii=False)

def main():
    train_data, dev_data, tag2int, int2tag = load_data()
    tokenizer = load_tokenizer()
    train_data_loader = create_dataloader(
        train_data, tokenizer, tag2int, training=True, batch_size=train_batch_size)
    dev_data_loader = create_dataloader(
        dev_data, tokenizer, tag2int, training=False, batch_size=eval_batch_size)
    model = load_model(num_labels=len(tag2int))

    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = len(train_data_loader) * num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)

    trainer = IntentClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data_loader=train_data_loader,
        dev_data_loader=dev_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tag2int=tag2int,
        int2tag=int2tag
    )
    trainer.train()


if __name__ == "__main__":
    main()
