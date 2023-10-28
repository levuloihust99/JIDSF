import os
import logging
import time
from tqdm import tqdm
import datetime
import json
import random
import string
import torch

from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from .configuration import NERConfig
from utils.utils import setup_logger

logger = logging.getLogger(__name__)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class NERTrainer(object):
    def __init__(
        self,
        train_dataloader,
        dev_dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
        config: NERConfig,
        tag2int,
        int2tag
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(config.gpu_id))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}, {}'.format(torch.cuda.get_device_name(config.gpu_id), torch.cuda.get_device_capability(config.gpu_id)))
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("MPS backend is available, using MPS.")
        else:
            logger.info('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.tag2int = tag2int
        self.int2tag = int2tag

        run_id = ''.join(random.choice(string.digits + string.ascii_uppercase) for _ in range(16))
        self.checkpoint_dir = os.path.join(
            config.model_save,
            config.data,
            os.path.basename(config.model_path),
            run_id
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        with open(os.path.join(self.checkpoint_dir, "label_mappings.json"), "w") as writer:
            json.dump(tag2int, writer, indent=4, ensure_ascii=False)
        with open(os.path.join(self.checkpoint_dir, "training_config.json"), "w") as writer:
            json.dump(config.__dict__, writer, indent=4, ensure_ascii=False)

        setup_logger(logger, log_file=os.path.join(self.checkpoint_dir, 'track.log'))

    def train(self):
        self.best_result = 0
        self.best_report = None
        
        self.model.train()
        global_step = 0
        for epoch in range(0, self.config.num_train_epochs):
            logger.info(' Epoch {:} / {:}'.format(epoch + 1, self.config.num_train_epochs))

            t0 = time.time()
            total_loss = 0
            self.model.train()
            progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))

            for step, batch in enumerate(self.train_dataloader):
                batch_input_ids = batch['input_ids'].to(self.device)
                batch_attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)

                batch_outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels if not self.config.custom_train else None,
                    return_dict=True
                )
                if self.config.custom_train:
                    entity_mask = batch['entity_mask'].to(self.device)
                    outside_mask = batch['outside_mask'].to(self.device)
                    is_outside = batch['is_outside'].to(self.device)

                    batch_size, seq_length = batch_labels.shape
                    is_outside_2d = torch.tile(torch.unsqueeze(is_outside, dim=-1), dims=(1, seq_length)) # batch_size x seq_length
                    batch_active_mask = is_outside_2d * outside_mask + (1 - is_outside_2d) * entity_mask # batch_size x seq_length
                    batch_logits = batch_outputs.logits # batch_size x seq_length x num_classes
                    _, _, num_classes = batch_logits.shape
                    batch_active_labels = []
                    batch_active_logits = []

                    for idx in range(batch_size):
                        active_mask = torch.tensor(batch_active_mask[idx], dtype=torch.bool)
                        active_labels = batch_labels[idx][active_mask]
                        active_logits = batch_logits[idx][active_mask]
                        if self.config.ignore_index == 0:
                            active_labels = active_labels[1:-1] # ignore [CLS], [SEP]
                            active_logits = active_logits[1:-1] # ignore [CLS], [SEP]
                        batch_active_labels.append(active_labels)
                        batch_active_logits.append(active_logits)
                    
                    batch_active_labels = torch.cat(batch_active_labels, dim=0)
                    batch_active_logits = torch.cat(batch_active_logits, dim=0)
                    batch_active_onehot_labels = torch.nn.functional.one_hot(batch_active_labels, num_classes=num_classes)
                    num_active_tokens = batch_active_labels.size(0)

                    batch_loss = (-batch_active_onehot_labels
                                * torch.nn.functional.log_softmax(batch_active_logits, dim=-1))
                    batch_loss = torch.sum(batch_loss) / num_active_tokens
                    total_loss += batch_loss
                else : 
                    batch_loss = batch_outputs.loss
                    total_loss += batch_loss.item()
                
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": float(batch_loss)})
                global_step += 1

                if isinstance(self.config.save_freq, int) and global_step % self.config.save_freq == 0:
                    if self.config.do_eval:
                        self.eval()
                    else:
                        output_dir = os.path.join(self.checkpoint_dir,
                                                'checkpoint-{}-{}-{}'.format(
                                                    self.model.__class__.__qualname__,
                                                    "step{:07d}-loss_{:.7f}".format(global_step, batch_loss.item()),
                                                    self.config.learning_rate
                                                ))
                        self.save_checkpoint(output_dir)

            avg_train_loss = total_loss / len(self.train_dataloader)

            logger.info("Average training loss: {0:.5f}".format(avg_train_loss))
            logger.info("Training epoch took: {:}".format(format_time(time.time() - t0)))
            if self.config.save_freq == "epoch":
                if self.config.do_eval:
                    self.eval()
                else:
                    output_dir = os.path.join(self.checkpoint_dir,
                                            'checkpoint-{}-{}-{}'.format(
                                                self.model.__class__.__qualname__,
                                                "epoch{:03d}".format(epoch),
                                                self.config.learning_rate
                                            ))
                    self.save_checkpoint(output_dir)

        logger.info("\nTraining complete!\n*************************************************************************************************************************\n\n")
        with open(os.path.join(self.checkpoint_dir, "best_report.txt"), "w") as writer:
            writer.write(self.best_report)

    def save_checkpoint(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model to %s" % output_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "label_mappings.json"), "w") as writer:
            json.dump(self.tag2int, writer, indent=4, ensure_ascii=False)
        with open(os.path.join(output_dir, "training_config.json"), "w") as writer:
            json.dump(self.config.__dict__, writer, indent=4, ensure_ascii=False)
            
    def eval(self):
        logger.info("Running Validation...")
        self.model.eval()
        t0 = time.time()

        y_true = []
        y_pred = []

        for batch in self.dev_dataloader:
            batch = {k : v.to(self.device) for k, v in batch.items()}
            batch_input_ids = batch["input_ids"]
            batch_attention_mask = batch["attention_mask"]
            batch_label_ids = batch["labels"]

            with torch.no_grad():
                batch_outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_dict=True
                )

            batch_logits = batch_outputs.logits
            batch_size, *_ = batch_logits.shape
            batch_preds = torch.argmax(batch_logits, dim=2)

            for idx in range(batch_size):
                active_mask = torch.tensor(batch_attention_mask[idx], dtype=torch.bool)
                true_label = batch_label_ids[idx][active_mask][1:-1]
                pred_label = batch_preds[idx][active_mask][1:-1]
                y_true.append([self.int2tag[int(truth)] for truth in true_label])
                y_pred.append([self.int2tag[int(pred)] for pred in pred_label])

        report = classification_report(y_true, y_pred, digits=4)
        logger.info("Validation took: {:}".format(format_time(time.time() - t0)))

        metrics = {
            "slot_precision": precision_score(y_true, y_pred),
            "slot_recall": recall_score(y_true, y_pred),
            "slot_f1": f1_score(y_true, y_pred)
        }
        logger.info("Metrics: ", metrics)

        if self.best_result == 0 or self.best_result < metrics['slot_f1']:
            if self.config.save_checkpoints:
                output_dir = os.path.join(self.checkpoint_dir,
                                        'checkpoint-{}-{}-{:.6f}'.format(self.model.__class__.__qualname__,
                                                                        self.config.learning_rate,
                                                                        metrics['slot_f1']))
                logger.info("Saving model to %s" % output_dir)
                self.save_checkpoint(output_dir)

                output_eval_file = os.path.join(output_dir, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    writer.write(report)

                self.best_result = metrics['slot_f1']

            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            self.best_report = report

        self.model.train()
