import os
import json
import torch
import random
import logging

from tqdm import tqdm
from collections import defaultdict
from typing import Text, List, Dict, Any

from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class IntentGroupDataloader:
    def __init__(
        self,
        tokenizer = None,
        data: List[Dict[Text, Any]] = None,
        data_path: Text = None,
        bsz: int = None,
        name_id_mapping: Dict[Text, int] = None,
        max_seq_len: int = 512
    ):
        self.data = data
        if not data:
            self.load_data(data_path)
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.tokenize_dataset()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.name_id_mapping = name_id_mapping
        self.id_name_mapping = None
        self.intent_ids = []
        self.intent_names = []
        self.group_intent()
    
    def load_data(self, data_path):
        data = []
        with open(data_path) as reader:
            for line in reader:
                data.append(json.loads(line.strip()))
        self.data = data
    
    def tokenize_dataset(self):
        for item in tqdm(self.data, desc="Tokenizing"):
            tokens = self.tokenizer.tokenize(item["sentence"])
            item["input_ids"] = self.tokenizer.convert_tokens_to_ids(tokens)
    
    def group_intent(self):
        intent_tracker = defaultdict(list)
        for item in self.data:
            intent_tracker[item["intent"]].append(item)
        self.intent_group = intent_tracker
        for i, name in enumerate(tqdm(self.intent_group, desc="Grouping")):
            self.intent_ids.append(i)
            self.intent_names.append(name)
        if not self.name_id_mapping:
            self.name_id_mapping = {intent: idx for idx, intent in enumerate(self.intent_names)}
        self.id_name_mapping = {v: k for k, v in self.name_id_mapping.items()}
        if self.bsz is None:
            self.bsz = len(intent_tracker)
        if self.bsz > len(intent_tracker):
            logger.warning(
                "Cannot set batch size larger than the number of intent. "
                "Number of intent = {n_intent}. Batch size = {bsz}. "
                "Set the batch size to the number of intent {n_intent}"
                .format(n_intent=len(intent_tracker), bsz=self.bsz)
            )
            self.bsz = len(intent_tracker)

    def __iter__(self):
        while True:
            sampled_intents = random.sample(self.intent_names, self.bsz)
            batch_labels = []
            batch_input_ids = []
            for intent in sampled_intents:
                per_intent_samples = random.sample(self.intent_group[intent], 2)
                per_intent_input_ids = [sample["input_ids"] for sample in per_intent_samples]
                batch_input_ids.extend(per_intent_input_ids)
                batch_labels.append(self.name_id_mapping[intent])
            
            max_seq_len = min(self.max_seq_len, max(len(token_ids) for token_ids in batch_input_ids))
            padded_batch_input_ids = []
            padded_batch_attn_mask = []
            for token_ids in batch_input_ids:
                if len(token_ids) > max_seq_len - 2:
                    token_ids = token_ids[:max_seq_len - 2]
                token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
                padding_len = max_seq_len - len(token_ids)
                attn_mask = [1] * len(token_ids) + [0] * padding_len
                token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_len
                padded_batch_input_ids.append(token_ids)
                padded_batch_attn_mask.append(attn_mask)
            
            batch_input_tensor = (
                torch.tensor(padded_batch_input_ids)
                        .view(self.bsz, 2, -1)
                        .transpose(0, 1)
                        .contiguous()
                        .view(self.bsz * 2, -1)
                        .to(self.device)
            )
            batch_attn_mask_tensor = (
                torch.tensor(padded_batch_attn_mask)
                        .view(self.bsz, 2, -1)
                        .transpose(0, 1)
                        .contiguous()
                        .view(self.bsz * 2, -1)
                        .to(self.device)
            )
            yield {
                "input_ids": batch_input_tensor,
                "attention_mask": batch_attn_mask_tensor,
                "labels": torch.tensor(batch_labels).to(self.device)
            }


def get_collate_fn(tokenizer, name_id_mapping, max_seq_len: int = 512):
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
            max_length=max_seq_len,
            return_tensors="pt"
        )

        labels = torch.tensor([name_id_mapping[intent] for intent in intents], dtype=torch.int64)
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            "texts": texts,
            "intents": intents
        }
    return collate_fn


def create_eval_dataloader(
    data: List[Dict[Text, Any]],
    tokenizer, name_id_mapping,
    batch_size: int,
):
    dataset = Dataset.from_list(data)
    collate_fn = get_collate_fn(tokenizer, name_id_mapping)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


def main():
    from transformers import BertTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import BertConverter
    data_path = "final/data/train_final_20230919.jsonl"
    tokenizer_path = "NlpHUST/vibert4news-base-cased"

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    converter = BertConverter(tokenizer)
    fast_tokenizer = converter.converted()
    cache_path = os.path.join(".cache/NlpHUST-fast-tokenizer")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast_tokenizer,
        unk_token=tokenizer.unk_token,
        pad_token=tokenizer.pad_token,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        mask_token=tokenizer.mask_token
    )
    fast_tokenizer.save_pretrained(cache_path)

    dataloader = IntentGroupDataloader(
        data_path = data_path,
        tokenizer=fast_tokenizer
    )

    iterator = iter(dataloader)
    sample = next(iterator)
    print("done")


if __name__ == "__main__":
    main()