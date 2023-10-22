import os
import json
import time
import torch
import logging

from tqdm import tqdm
from typing import Text
from itertools import chain
from collections import defaultdict

from finalround.configuration.intent_cls import IntentClassifierConfig
from finalround.tokenization.resolver import resolve_tokenizer
from finalround.modeling.intent_cls_resolver import resolve_intent_cls_model
from finalround.dataloader.intent_group_dataloader import (
    IntentGroupDataloader,
    create_eval_dataloader
)
from finalround.optimization import create_optimizer_and_scheduler
from finalround.losses.intent_cls import InbatchLossCalculator

logger = logging.getLogger(__name__)


def load_data(data_path: Text):
    data = []
    with open(data_path) as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    return data


def f_score(precision: float, recall: float):
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


class IntentClassifierTrainer:
    def __init__(
        self,
        config: IntentClassifierConfig,
    ):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # data
        data = load_data(config.data_path)
        with open(config.train_indices_path, "r") as reader:
            train_indices = json.load(reader)

        train_data = []
        for idx in train_indices:
            train_data.append(data[idx])

        all_intents = []
        for item in data:
            if item["intent"] not in all_intents:
                all_intents.append(item["intent"])
        name_id_mapping = {intent: idx for idx, intent in enumerate(all_intents)}

        # tokenizer and model
        self.tokenizer = resolve_tokenizer(config.tokenizer_type, config.tokenizer_path)
        self.model = resolve_intent_cls_model(
            config.model_type, config.model_path, num_labels=len(all_intents), add_pooling_layer=config.add_pooling_layer)

        self.dataloader = IntentGroupDataloader(
            data=train_data,
            tokenizer=self.tokenizer,
            bsz=config.train_batch_size,
            name_id_mapping=name_id_mapping,
            max_seq_len=config.max_seq_length
        )

        if self.config.do_eval:
            with open(self.config.dev_indices_path, "r") as reader:
                dev_indices = json.load(reader)
            dev_data = []
            for idx in dev_indices:
                dev_data.append(data[idx])
            self.eval_dataloader = create_eval_dataloader(
                dev_data, self.tokenizer, name_id_mapping, batch_size=config.eval_batch_size)
        
        # create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=self.model,
            total_steps=config.total_updates,
            weight_decay=config.weight_decay,
            warmup_proportion=config.warmup_proportion,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        # loss calculator
        self.loss_calculator = InbatchLossCalculator(
            intent_embs=self.model.classifier.weight,
            metrics=config.sim_func,
            scale_factor=config.scale_cosine_factor
        )

        self.model.to(self.device)

    def train(self):
        logger.info("Start training...")
        self.best_result = 0.0

        self.model.train()
        data_iterator = iter(self.dataloader)
        progress_bar = tqdm(total=self.config.total_updates, desc="Iteration")
        encoder = self.model.get_encoder()
        for iteration in range(self.config.total_updates):
            batch = next(data_iterator)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits, pooled_output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = self.loss_calculator.calculate(logits, pooled_output, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update(1)

            if self.config.do_eval:
                if (iteration + 1) % self.config.eval_steps == 0:
                    self.eval()
            else:
                if (iteration + 1) % self.config.save_steps == 0:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.checkpoint_dir,
                            "checkpoint-{}-{}-{}".format(
                                self.model.__class__.__qualname__,
                                self.config.learning_rate,
                                "step{:07d}-loss_{:.7f}".format(iteration + 1, loss.item()),
                            )
                        )
                    )


    def eval(self):
        self.model.eval()
        logger.info("Running evaluation...")
        t0 = time.time()

        eval_tracker = {
            intent_id: {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "Support": 0
            } for intent, intent_id in self.dataloader.name_id_mapping.items()
        }
        eval_progress_bar = tqdm(desc="Eval batch", total=len(self.eval_dataloader))
        for batch in self.eval_dataloader:
            batch_input_ids = batch["input_ids"].to(self.device)
            batch_attention_mask = batch["attention_mask"].to(self.device)
            batch_labels = batch["labels"].to(self.device)

            with torch.no_grad():
                batch_logits, _ = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
            batch_preds = torch.argmax(batch_logits, dim=-1)

            for pred, label in zip(batch_preds, batch_labels):
                pred = pred.item()
                label = label.item()
                if pred != label:
                    eval_tracker[pred]["FP"] += 1
                    eval_tracker[label]["FN"] += 1
                else:
                    eval_tracker[pred]["TP"] += 1
                eval_tracker[label]["Support"] += 1
            
            eval_progress_bar.update(1)
        
        eval_results = {}
        # calculate metrics for each class
        for intent_id, metrics in eval_tracker.items():
            if metrics["TP"] == 0:
                precision = 0
                recall = 0
            else:
                precision = metrics["TP"] / (metrics["TP"] + metrics["FP"])
                recall = metrics["TP"] / (metrics["TP"] + metrics["FN"])
            f1 = f_score(precision, recall)
            eval_results[self.dataloader.id_name_mapping[intent_id]] = {
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "Support": metrics["Support"]
            }

        # prepare metrics array of all classes
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

        # macro average
        macro_p = sum(precisions) / len(precisions)
        macro_r = sum(recalls) / len(recalls)
        macro_f = f_score(macro_p, macro_r)
        macro_eval = {
            "Precision": macro_p,
            "Recall": macro_r,
            "F1-score": macro_f,
            "Support": total
        }

        # weighted average
        weights = [s / total for s in supports]
        weighted_p = 0.0
        weighted_r = 0.0
        for w, p, r in zip(weights, precisions, recalls):
            weighted_p += w * p
            weighted_r += w * r
        weighted_f = f_score(weighted_p, weighted_r)
        weighted_eval = {
            "Precision": weighted_p,
            "Recall": weighted_r,
            "F1-score": weighted_f,
            "Support": total
        }

        # micro average
        sum_TP = 0
        sum_FP = 0
        sum_FN = 0
        for metrics in eval_tracker.values():
            sum_TP += metrics["TP"]
            sum_FP += metrics["FP"]
            sum_FN += metrics["FN"]
        micro_p = sum_TP / (sum_TP + sum_FP)
        micro_r = sum_TP / (sum_TP + sum_FN)
        micro_f = f_score(micro_p, micro_r)
        micro_eval = {
            "Precision": micro_p,
            "Recall": micro_r,
            "F1-score": micro_f,
            "Support": total
        }

        # format eval results
        metrics_iterator = list(chain(
            [("", {"Precision": "Precision", "Recall": "Recall", "F1-score": "F1-score", "Support": "Support"})],
            eval_results.items(), 
            [
                (None, {"Precision": None, "Recall": None, "F1-score": None, "Support": None}),
                ("Micro", micro_eval),
                ("Macro", macro_eval),
                ("Weighted", weighted_eval)
            ]
        ))
        cols_length = defaultdict(list)
        cols_value = defaultdict(list)
        for idx, (name, metrics) in enumerate(metrics_iterator):
            if name is None:
                cols_value["name"].append(None)
                for metric_name in metrics:
                    cols_value[metric_name].append(None)
                continue
            cols_length["name"].append(len(name))
            cols_value["name"].append(name)
            for metric_name, metric_value in metrics.items():
                if idx == 0:
                    formatted_metric_value = metric_value
                elif metric_name != "Support":
                    formatted_metric_value = "{:.4f}".format(metric_value)
                else:
                    formatted_metric_value = "{}".format(metric_value)
                cols_length[metric_name].append(len(formatted_metric_value))
                cols_value[metric_name].append(formatted_metric_value)

        max_cols_length = {k: max(v) for k, v in cols_length.items()}
        eval_formatted_result = ""
        for idx, (name, metrics) in enumerate(metrics_iterator):
            if name is None:
                eval_formatted_result += "\n"
            else:
                eval_formatted_result += "{{:>{}s}}".format(max_cols_length["name"]).format(name)
                for metric_name in metrics:
                    eval_formatted_result += (
                        "{{:>{}s}}"
                        .format(max_cols_length[metric_name] + 5)
                        .format(cols_value[metric_name][idx])
                    )
                eval_formatted_result += "\n"
        logger.info("***** Eval results *****")
        logger.info("\n\n%s", eval_formatted_result)

        # save checkpoint
        eval_score = None
        if self.config.eval_metric == "micro":
            eval_score = micro_eval["F1-score"]
        elif self.config.eval_metric == "macro":
            eval_score = macro_eval["F1-score"]
        elif self.config.eval_metric == "weighted":
            eval_score = weighted_eval["F1-score"]
        else:
            raise Exception("Unknown eval metric '{}'".format(self.config.eval_metric))
        output_dir = os.path.join(
            self.config.checkpoint_dir,
            "checkpoint-{}-{}-{}".format(
                self.model.__class__.__qualname__,
                self.config.learning_rate,
                "{}_f1_{:.6f}".format(self.config.eval_metric, eval_score)
            )
        )
        if self.config.only_save_better:
            if self.best_result == 0 or self.best_result < eval_score:
                self.best_result = eval_score
                self.save_checkpoint(output_dir)
                output_eval_file = os.path.join(output_dir, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    writer.write(eval_formatted_result)
        else:
            self.save_checkpoint(output_dir)
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(eval_formatted_result)

        self.model.train()

    def save_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        logger.info("Saving model to %s" % checkpoint_dir)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        with open(os.path.join(checkpoint_dir, "label_mappings.json"), "w") as writer:
            json.dump(self.dataloader.name_id_mapping, writer, indent=4, ensure_ascii=False)
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as writer:
            json.dump(self.config.to_json(), writer, indent=4, ensure_ascii=False)
