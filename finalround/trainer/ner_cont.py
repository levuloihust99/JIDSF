import os
import json
import time
import torch
import logging
import datetime

from tqdm import tqdm
from typing import Text

from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from finalround.configuration.ner_cont import NERContConfig
from finalround.tokenization.resolver import resolve_tokenizer
from finalround.modeling.ner_resolver import resolve_ner_cont_model
from finalround.dataloader.ner_cont_dataloader import (
    NERContDataloader, NERSequenceDataloader
)
from finalround.optimization import create_optimizer_and_scheduler
from finalround.losses.ner_cont import NERContLoss

logger = logging.getLogger(__name__)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_data(data_path: Text):
    data = []
    with open(data_path) as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    for idx, item in enumerate(data):
        item["idx"] = idx
    return data


class NERContTrainer:
    def __init__(self, config: NERContConfig):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # data
        data = load_data(config.data_path)
        with open(config.train_indices_path, "r") as reader:
            train_indices = json.load(reader)
        
        train_data = []
        for idx in train_indices:
            train_data.append(data[idx])
        
        all_entities = []
        for idx, item in enumerate(data):
            item["idx"] = idx
            token_labels = item["labels"]
            for label in token_labels:
                if label != "O":
                    label = label[2:]
                if label not in all_entities:
                    all_entities.append(label)
        
        all_tags = []
        for entity in all_entities:
            if entity != "O":
                all_tags.extend([f"B-{entity}", f"I-{entity}"])
            else:
                all_tags.append(entity)
        all_tags = ["-PAD-"] + all_tags
        
        name_id_mapping = {tag: idx for idx, tag in enumerate(all_tags)}
        id_name_mapping = {idx: tag for idx, tag in enumerate(all_tags)}
        self.name_id_mapping = name_id_mapping
        self.id_name_mapping = id_name_mapping

        # tokenizer and model
        self.tokenizer = resolve_tokenizer(config.tokenizer_type, config.tokenizer_path)
        self.model = resolve_ner_cont_model(
            config.model_type, config.model_path, num_labels=len(name_id_mapping), add_pooling_layer=config.add_pooling_layer
        )

        self.dataloader = NERContDataloader(
            data=train_data,
            tokenizer=self.tokenizer,
            name_id_mapping=name_id_mapping,
            max_seq_len=config.max_seq_length
        )

        if self.config.do_eval:
            with open(self.config.dev_indices_path, "r") as reader:
                dev_indices = json.load(reader)
            dev_data = []
            for idx in dev_indices:
                dev_data.append(data[idx])
            self.dev_dataloader = NERSequenceDataloader(
                data=dev_data,
                tokenizer=self.tokenizer,
                name_id_mapping=name_id_mapping,
                max_seq_len=config.max_seq_length,
                training=False
            )
        
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
        self.loss_calculator = NERContLoss(
            label_embeddings=self.model.label_embeddings,
            metrics=config.sim_func,
            scale_factor=config.scale_cosine_factor,
            ignore_index=config.ignore_index,
            pad_label_id=name_id_mapping["-PAD-"]
        )

        self.model.to(self.device)

    def train(self):
        logger.info("Start training...")
        self.best_result = 0
        data_iterator = iter(self.dataloader)
        progress_bar = tqdm(total=self.config.total_updates, desc="Step")
        for step in range(self.config.total_updates):
            batch = next(data_iterator)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            token_embeddings = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            active_mask = batch["attention_mask"].view(-1) == 1
            active_embeddings = token_embeddings.view(-1, self.model.config.hidden_size)[active_mask]
            active_labels = batch["labels"].view(-1)[active_mask]
            loss = self.loss_calculator.calculate(active_embeddings, active_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update(1)

            if self.config.do_eval:
                if (step + 1) % self.config.eval_steps == 0:
                    self.eval()
            else:
                if (step + 1) % self.config.save_steps == 0:
                    self.save_checkpoint(
                        self.config.checkpoint_dir,
                        "checkpoint-{}-{}-{}".format(
                            self.model.__class__.__qualname__,
                            "step{:07d}-loss_{:.7f}".format(step + 1, loss.item())
                        )
                    )

    def eval(self):
        self.model.eval()
        logger.info("Running evaluation...")
        t0 = time.time()

        y_true = []
        y_pred = []

        for batch in self.dev_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                sequence_output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
            batch_logits = torch.matmul(sequence_output, self.model.label_embeddings)
            batch_preds = torch.argmax(batch_logits, dim=-1)
            
            batch_size = batch_preds.size(0)
            for idx in range(batch_size):
                active_mask = batch["attention_mask"][idx].to(torch.bool)
                true_label = batch["labels"][idx][active_mask][1:-1]
                pred_label = batch_preds[idx][active_mask][1:-1]
                y_true.append([self.id_name_mapping[truth.item()] for truth in true_label])
                y_pred.append([self.id_name_mapping[pred.item()] for pred in pred_label])
        
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("***** Eval results *****")
        logger.info("\n\n%s", report)
        logger.info("Validation took: {}".format(format_time(time.time() - t0)))

        eval_metrics = {
            "micro": f1_score(y_true, y_pred, average="micro"),
            "macro": f1_score(y_true, y_pred, average="macro"),
            "weighted": f1_score(y_true, y_pred, average="weighted")
        }
        eval_score = eval_metrics[self.config.eval_metric]
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
                    writer.write(report)
        else:
            self.save_checkpoint(output_dir)
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(report)
        
        self.model.train()

    def save_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logger.info("Saving model to %s" % checkpoint_dir)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        with open(os.path.join(checkpoint_dir, "label_mappings.json"), "w") as writer:
            json.dump(self.name_id_mapping, writer, indent=4, ensure_ascii=False)
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as writer:
            json.dump(self.config.to_json(), writer, indent=4, ensure_ascii=False)
