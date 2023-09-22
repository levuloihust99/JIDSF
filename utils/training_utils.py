import json
import pandas as pd
from typing import Dict, Text, Literal
import logging
from tqdm import tqdm
from datasets import Dataset

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


from transformers import (
    BertTokenizer,
    PhobertTokenizer,
    ElectraTokenizer,
    AutoTokenizer,
    ElectraConfig,
    DataCollatorForTokenClassification
)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model.configuration import NERConfig
from model.tokenization import SentencePieceTokenizer
from model.modeling import (
    BertPosTagger,
    BertPosTaggerElectra,
    PhoBertPosTagger,
    XLMRobertaPosTagger,
    DistilBertPosTagger
)

import sentencepiece as spm

logger = logging.getLogger(__name__)


def _load_data(config: NERConfig, format: Literal["csv", "jsonlines"] = "csv"):
    logger.info("Loading data...")
    # load train sentences
    if format == "csv":
        train_data_path = config.path_to_data['train']
        train_df = pd.read_csv(train_data_path, header=0)
        if config.use_word_segmenter:
            train_tokens = train_df.segmented_tokens
            train_labels = train_df.segmented_labels
        else:
            train_tokens = train_df.tokens
            train_labels = train_df.labels
        train_sentences = [list(zip(eval(tokens), eval(labels))) for tokens, labels in zip(train_tokens, train_labels)]
    else:
        if config.use_word_segmenter:
            token_field = "segmented_tokens"
            label_field = "segmented_labels"
        else:
            token_field = "tokens"
            label_field = "labels"
        all_sentences = []
        with open(config.path_to_data["all"], "r") as reader:
            for line in reader:
                item = json.loads(line.strip())
                all_sentences.append(list(zip(item[token_field], item[label_field])))
        with open(config.path_to_data["train_indices"], "r") as reader:
            train_indices = json.load(reader)
        train_sentences = []
        for idx in train_indices:
            train_sentences.append(all_sentences[idx])
    
    if not config.do_eval:
        return train_sentences, []

    if format == "csv":
        dev_data_path   = config.path_to_data['dev']
        dev_df   = pd.read_csv(dev_data_path, header=0)
        if config.use_word_segmenter:
            dev_tokens = dev_df.segmented_tokens
            dev_labels = dev_df.segmented_labels
        else:
            dev_tokens = dev_df.tokens
            dev_labels = dev_df.labels
        dev_sentences = [list(zip(eval(tokens), eval(labels))) for tokens, labels in zip(dev_tokens, dev_labels)]
    else:
        with open(config.path_to_data["dev_indices"], "r") as reader:
            dev_indices = json.load(reader)
        dev_sentences = []
        for idx in dev_indices:
            dev_sentences.append(all_sentences[idx])

    return train_sentences, dev_sentences


def load_data(config: NERConfig):
    train_sentences, dev_sentences = _load_data(config, format=config.data_format)
    tags = set([item for sublist in train_sentences + dev_sentences for _, item in sublist])
    processed_tags = set()
    for tag in tags: # this guarantees both B- and I- are included.
                     # In some cases, there may be a B- tag but no I- tag.
        if tag == 'O' and tag != 'DEFAULT':
            processed_tags.add(tag)
        elif tag != 'DEFAULT':
            processed_tags.add('B-' + tag[2:])
            processed_tags.add('I-' + tag[2:])
    tags = sorted(processed_tags)
    tags.reverse()
    logger.info("All tags...")
    logger.info(tags)

    tag2int = {}
    int2tag = {}
    for i, tag in enumerate(tags):
        tag2int[tag] = i + 1
        int2tag[i + 1] = tag
    tag2int['-PAD-'] = 0
    tag2int['DEFAULT'] = 0
    int2tag[0] = '-PAD-'

    return train_sentences, dev_sentences, tag2int, int2tag
    

def create_dataloader(
    config: NERConfig,
    tagged_sequences,
    tokenizer,
    tag2int: Dict,
    training=True,
):
    """Create Pytorch DataLoader from a list of tagged sequences.

    Examples of a tagged sequence:
        [('Thành', 'B-LOCATION'), ('phố', 'I-LOCATION'), ('Hà', 'I-LOCATION'), ('Nội', 'I-LOCATION')]
    """
    all_input_ids = []
    all_attention_mask = []
    all_label_ids = []
    all_is_outside = []
    all_entity_mask = []
    all_outside_mask = []

    logger.info("Creating dataloader for {} data...".format('train' if training else 'dev'))
    for sequence in tqdm(tagged_sequences):
        out_tokens = []
        out_labels = []
        is_outside = False

        for token, label in sequence:
            if label == 'DEFAULT':
                is_outside = True
            subword_tokens = tokenizer.tokenize(token)
            out_tokens.extend(subword_tokens)
            
            if label.startswith('B-'):
                out_labels.extend([label] + ['I-' + label[2:]] * (len(subword_tokens) - 1))
            else:
                out_labels.extend([label] * len(subword_tokens))
        
        all_is_outside.append(is_outside)

        if len(out_tokens) > config.max_seq_length - 2:
            out_tokens = out_tokens[:config.max_seq_length - 2]
            out_labels = out_labels[:config.max_seq_length - 2]
        
        out_tokens = [tokenizer.cls_token] + out_tokens + [tokenizer.sep_token]
        if not is_outside:
            entity_mask = [1] + [0 if lb == 'O' else 1 for lb in out_labels] + [1]
            outside_mask = [0] * len(out_tokens)
        else:
            entity_mask = [0] * len(out_tokens)
            outside_mask = [1] + [1 if lb == 'O' else 0 for lb in out_labels] + [1]
        out_labels = ['-PAD-'] + out_labels + ['-PAD-']
        attention_mask = [1] * len(out_tokens)

        out_token_ids = tokenizer.convert_tokens_to_ids(out_tokens)
        out_label_ids = [tag2int[label] for label in out_labels]
    
        all_input_ids.append(out_token_ids)
        all_attention_mask.append(attention_mask)
        all_label_ids.append(out_label_ids)
        all_entity_mask.append(entity_mask)
        all_outside_mask.append(outside_mask)

    dataset = pd.DataFrame(
        {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_label_ids,
            'is_outside': all_is_outside,
            'entity_mask': all_entity_mask,
            'outside_mask': all_outside_mask
        }
    )
    dataset = Dataset.from_pandas(dataset)
    collate_fn = get_collate_fn(tokenizer.pad_token_id)
   
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size if training else config.eval_batch_size,
        shuffle=training,
        collate_fn=collate_fn # padding with max_seq_length in batch_size
    )
    return dataloader


def get_collate_fn(pad_token_id):
    def collate_fn(items):
        batch_input_ids = [item['input_ids'] for item in items]
        seq_lengths = [len(input_ids) for input_ids in batch_input_ids]
        max_seq_length = max(seq_lengths)

        batch_input_ids = []
        batch_attention_mask = []
        batch_entity_mask = []
        batch_outside_mask = []
        batch_is_outside = []
        batch_labels = []

        for item in items:
            num_padding = max_seq_length - len(item['input_ids'])

            item['input_ids'] = item['input_ids'] + [pad_token_id] * num_padding
            batch_input_ids.append(item['input_ids'])

            item['attention_mask'] = item['attention_mask'] + [0] * num_padding
            batch_attention_mask.append(item['attention_mask'])
    
            item['labels'] = item['labels'] + [0] * num_padding
            batch_labels.append(item['labels'])

            item['entity_mask'] = item['entity_mask'] + [0] * num_padding
            batch_entity_mask.append(item['entity_mask'])

            item['outside_mask'] = item['outside_mask'] + [0] * num_padding
            batch_outside_mask.append(item['outside_mask'])

            batch_is_outside.append(item['is_outside'])
        
        ret = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.int64),
            "labels": torch.tensor(batch_labels, dtype=torch.int64),
            "is_outside": torch.tensor(batch_is_outside, dtype=torch.int64),
            "entity_mask": torch.tensor(batch_entity_mask, dtype=torch.float32),
            "outside_mask": torch.tensor(batch_outside_mask, dtype=torch.float32)
        }
        return ret

    return collate_fn


def load_tokenizer(config: NERConfig, **kwargs):
    logger.info('Loading BERT tokenizer...')
    if config.tokenizer_path.find('NlpHUST/vibert4news') > -1:
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path, **kwargs)
    elif config.tokenizer_path.find('NlpHUST/vi-electra') > -1:
        tokenizer = ElectraTokenizer.from_pretrained(config.tokenizer_path, **kwargs)
    elif config.tokenizer_path.find('vinai/phobert') > -1:  #
        tokenizer = PhobertTokenizer.from_pretrained(config.tokenizer_path, **kwargs)
    elif config.tokenizer_path.find('xlm-roberta') > -1:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, **kwargs)
    elif config.tokenizer_path.find('distilbert') > -1:
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path, **kwargs)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path, **kwargs)

    return tokenizer


def load_model(config: NERConfig, num_classes: int):
    if config.model_path.find('NlpHUST/vibert4news') > -1:
        model = BertPosTagger.from_pretrained(
            config.model_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True,
            args=config,
        )

    # vi-electra-base
    elif config.model_path.find('NlpHUST/vi-electra') > -1:
        config = ElectraConfig.from_pretrained(config.model_path, finetuning_task='ner', num_labels=num_classes,
                                                output_attentions=False, output_hidden_states=True, )
        model = BertPosTaggerElectra.from_pretrained(
            config.model_path,
            config=config,
            args=config,
        )

    # PhoBert
    elif config.model_path.find('vinai/phobert') > -1:
        model = PhoBertPosTagger.from_pretrained(
            config.model_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True,
            args=config
        )

    # XLM-roberta
    elif config.model_path.find('xlm-roberta') > -1:
        model = XLMRobertaPosTagger.from_pretrained(
            config.model_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True,
            args=config
        )

    # DistilBert
    elif config.model_path.find('distilbert') > -1:
        model = DistilBertPosTagger.from_pretrained(
            config.model_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True,
            args=config
        )

    else:
        model = BertPosTagger.from_pretrained(
            config.model_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True,
            args=config,
        )

    return model


def create_optimizer_and_scheduler(model, config, total_steps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    num_warmup_steps_by_ratio = int(total_steps * config.warmup_proportion)
    num_warmup_steps_absolute = config.warmup_steps
    if num_warmup_steps_absolute == 0 or num_warmup_steps_by_ratio == 0:
        num_warmup_steps = max(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    else:
        num_warmup_steps = min(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler