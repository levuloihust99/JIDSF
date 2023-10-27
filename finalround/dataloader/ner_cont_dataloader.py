import os
import json
import torch
import random
import logging

from tqdm import tqdm
from collections import defaultdict
from typing import Text, List, Dict, Any, Union

from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class NERContDataloader:
    def __init__(
        self,
        tokenizer = None,
        data: Union[Text, List[Dict[Text, Any]]] = None,
        name_id_mapping: Dict[Text, int] = None,
        max_seq_len: int = 512
    ):
        self.data = data
        if isinstance(data, str):
            self.load_data(data)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.name_id_mapping = name_id_mapping
        self.id_name_mapping = {v: k for k, v in name_id_mapping.items()}
        self.tokenize_dataset()
        self.group_entities()

    def load_data(self, data_path):
        data = []
        with open(data_path) as reader:
            for line in reader:
                data.append(json.loads(line.strip()))
        for idx, item in enumerate(data):
            item["idx"] = idx
        self.data = data
    
    def tokenize_dataset(self):
        for item in tqdm(self.data, desc="Tokenizing"):
            out_tokens = []
            out_labels = []
            for token, label in zip(item["tokens"], item["labels"]):
                subword_tokens = self.tokenizer.tokenize(token)
                out_tokens.extend(subword_tokens)

                if label.startswith("B-"):
                    out_labels.extend([label] + ["I-" + label[2:]] * (len(subword_tokens) - 1))
                else:
                    out_labels.extend([label] * len(subword_tokens))
            
            if len(out_tokens) > self.max_seq_len - 2:
                out_tokens = out_tokens[:self.max_seq_len - 2]
                out_labels = out_labels[:self.max_seq_len - 2]
            
            out_tokens = [self.tokenizer.cls_token] + out_tokens + [self.tokenizer.sep_token]
            out_labels = ["-PAD-"] + out_labels + ["-PAD-"]

            out_token_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
            out_label_ids = [self.name_id_mapping[label] for label in out_labels]

            item["input_ids"] = out_token_ids
            item["label_ids"] = out_label_ids

    def group_entities(self):
        entity_tracker = defaultdict(list)
        for item in self.data:
            token_labels = item["labels"]
            token_label_tracker = set()
            for label in token_labels:
                if label not in token_label_tracker:
                    entity_tracker[label].append(item)
                    token_label_tracker.add(label)
        self.entity_group = entity_tracker

    def __iter__(self):
        while True:
            batch_samples = []
            idx_tracker = set()
            for entity in self.entity_group:
                per_entity_samples = random.sample(self.entity_group[entity], 2)
                for sample in per_entity_samples:
                    if sample["idx"] not in idx_tracker:
                        idx_tracker.add(sample["idx"])
                        batch_samples.append(sample)

            max_seq_len = min(self.max_seq_len, max(len(item["input_ids"]) for item in batch_samples))
            padded_batch_input_ids = []
            padded_batch_attn_mask = []
            padded_batch_label_ids = []
            for sample in batch_samples:
                input_ids = sample["input_ids"]
                padding_len = max_seq_len - len(input_ids)

                attn_mask = [1] * len(input_ids) + [0] * padding_len
                padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
                padded_label_ids = sample["label_ids"] + [self.name_id_mapping["-PAD-"]] * padding_len

                padded_batch_input_ids.append(padded_input_ids)
                padded_batch_attn_mask.append(attn_mask)
                padded_batch_label_ids.append(padded_label_ids)
            
            batch_input_tensor = torch.tensor(padded_batch_input_ids).to(self.device)
            batch_attn_mask_tensor = torch.tensor(padded_batch_attn_mask).to(self.device)
            batch_label_ids_tensor = torch.tensor(padded_batch_label_ids).to(self.device)

            yield {
                "input_ids": batch_input_tensor,
                "attention_mask": batch_attn_mask_tensor,
                "labels": batch_label_ids_tensor
            }


class NERSequenceDataloader:
    def __init__(
        self,
        data: Union[Text, List[Dict[Text, Any]]] = None,
        tokenizer = None,
        name_id_mapping: Dict[Text, int] = None,
        bsz: int = 16,
        max_seq_len: int = 512,
        training: bool = False
    ):
        self.data = data
        if isinstance(data, str):
            self.load_data(data)
        self.tokenizer = tokenizer
        self.name_id_mapping = name_id_mapping
        self.max_seq_len = max_seq_len
        self.training = training
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.tokenize_dataset()
        dataset = Dataset.from_list(self.data)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=bsz,
            shuffle=training,
            collate_fn=self.collate_fn
        )
        
    def load_data(self, data_path):
        data = []
        with open(data_path) as reader:
            for line in reader:
                data.append(json.loads(line.strip()))
        for idx, item in enumerate(data):
            item["idx"] = idx
        self.data = data
    
    def tokenize_dataset(self):
        for item in tqdm(self.data, desc="Tokenizing"):
            out_tokens = []
            out_labels = []
            for token, label in zip(item["tokens"], item["labels"]):
                subword_tokens = self.tokenizer.tokenize(token)
                out_tokens.extend(subword_tokens)

                if label.startswith("B-"):
                    out_labels.extend([label] + ["I-" + label[2:]] * (len(subword_tokens) - 1))
                else:
                    out_labels.extend([label] * len(subword_tokens))

            if len(out_tokens) > self.max_seq_len - 2:
                out_tokens = out_tokens[:self.max_seq_len - 2]
                out_labels = out_labels[:self.max_seq_len - 2]
            
            out_tokens = [self.tokenizer.cls_token] + out_tokens + [self.tokenizer.sep_token]
            out_labels = ["-PAD-"] + out_labels + ["-PAD-"]

            out_token_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
            out_label_ids = [self.name_id_mapping[label] for label in out_labels]

            item["input_ids"] = out_token_ids
            item["label_ids"] = out_label_ids

    def collate_fn(self, items: List[Dict[Text, Any]]):
        batch_input_ids = []
        batch_label_ids = []
        for item in items:
            batch_input_ids.append(item["input_ids"])
            batch_label_ids.append(item["label_ids"])

        max_seq_len = min(self.max_seq_len, max(len(item["input_ids"]) for item in items))

        batch_input_ids = []
        batch_attn_mask = []
        batch_label_ids = []
        for item in items:
            padding_len = max_seq_len - len(item["input_ids"])
            input_ids = item["input_ids"] + [self.tokenizer.pad_token_id] * padding_len
            attn_mask = [1] * len(item["input_ids"]) + [0] * padding_len
            label_ids = item["label_ids"] + [self.name_id_mapping["-PAD-"]] * padding_len
            batch_input_ids.append(input_ids)
            batch_attn_mask.append(attn_mask)
            batch_label_ids.append(label_ids)
        
        return {
            "input_ids": torch.tensor(batch_input_ids).to(self.device),
            "attention_mask": torch.tensor(batch_attn_mask).to(self.device),
            "labels": torch.tensor(batch_label_ids).to(self.device)
        }

    def __iter__(self):
        return iter(self.dataloader)


def main():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")

    data = []
    with open("final/data/ner/all.jsonl", "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

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

    train_data = []
    with open("final/data/ner/train_indices.json", "r") as reader:
        train_indices = json.load(reader)
    for idx in train_indices:
        train_data.append(data[idx])

    dev_data = []
    with open("final/data/ner/dev_indices.json", "r") as reader:
        dev_indices = json.load(reader)
    for idx in dev_indices:
        dev_data.append(data[idx])

    dataloader = NERContDataloader(
        tokenizer=tokenizer,
        data=train_data,
        name_id_mapping=name_id_mapping,
        max_seq_len=512
    )

    data_iterator = iter(dataloader)
    batch = next(data_iterator)

    dev_dataloader = NERSequenceDataloader(
        tokenizer=tokenizer,
        data=dev_data,
        name_id_mapping=name_id_mapping,
        bsz=16,
        max_seq_len=512, 
        training=False
    )
    for batch in dev_dataloader:
        pass
    print("done")


if __name__ == "__main__":
    main()