import re
import torch
import requests
import logging

import copy
import unicodedata
import unidecode
import numpy as np
from typing import Literal

logger = logging.getLogger(__name__)




class NERProcessor(object):
    def __init__(
        self,
        tokenizer,
        model,
        label_mappings,
        segmenter,
        args
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.label_mappings = label_mappings
        self.inverse_label_mappings = {v : k for k, v in label_mappings.items()}
        self.segmenter = segmenter
        self.args = args

        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(args.gpu_id))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}, {}'.format(torch.cuda.get_device_name(args.gpu_id), torch.cuda.get_device_capability(args.gpu_id)))
        else:
            logger.info('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        self.model.eval()
    
    def segment(self, text):
        return self.segmenter.segment(text)

    def extract_entities(
        self,
        tokens,
        labels,
    ):
        assert len(tokens) == len(labels)
        if len(tokens) == 0:
            return []
        entities = []
        idx = 0
        current_label = None
        current_idxs = []
        prev_token_is_entity = False

        while True:
            label = labels[idx]
            if label.startswith('B-'):
                if prev_token_is_entity:
                    entities.append({
                        'entity': current_label,
                        'indexes': current_idxs
                    })
                    current_idxs = []
                current_label = label[2:]
                current_idxs.append(idx)
                prev_token_is_entity = True
                if idx == len(labels) - 1:
                    entities.append({
                        'entity': current_label,
                        'indexes': current_idxs
                    })
            elif label.startswith('I-'):
                if prev_token_is_entity and label[2:] == current_label:
                    current_idxs.append(idx)
                    if idx == len(labels) - 1:
                        entities.append({
                            'entity': current_label,
                            'indexes': current_idxs,
                        })
                else:
                    if prev_token_is_entity:
                        entities.append({
                            'entity': current_label,
                            'indexes': current_idxs
                        })
                        current_idxs = []
                        current_label = None
                        prev_token_is_entity = False
            else:
                if prev_token_is_entity:
                    entities.append({
                        'entity': current_label,
                        'indexes': current_idxs
                    })
                    current_idxs = []
                    current_label = None
                    prev_token_is_entity = False

            idx += 1
            if idx == len(labels):
                break

        out_entities = []
        for entity in entities:
            entity_tokens = [tokens[idx] for idx in entity["indexes"]]
            entity_token_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
            entity_value = self.tokenizer.decode(entity_token_ids)
            out_entities.append({
                "entity": entity["entity"],
                "value": entity_value
            })
        return out_entities

    def get_prediction(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.forward(input_ids=input_ids)

        sequence_output = outputs[0]
        sequence_output = torch.squeeze(sequence_output, dim=0)
        
        predict_labels = torch.argmax(sequence_output, dim=-1)
        predict_labels = predict_labels.cpu().numpy().tolist()
        predict_labels = [self.inverse_label_mappings[label_id] for label_id in predict_labels]

        return tokens[1:-1], predict_labels[1:-1] # ignore special tokens

    def extract(self, text):
        if self.segmenter:
            input_text = self.segment(text)
        else:
            input_text = text
        if self.args.lower:
            input_text = input_text.lower()
        tokens, labels = self.get_prediction(input_text)
        entities = self.extract_entities(tokens, labels)
        if self.segmenter:
            for entity in entities:
                entity["value"] = entity["value"].replace("_", " ")
        return entities

    def extract_raw(self, text):
        if self.segmenter:
            input_text = self.segment(text)
        else:
            input_text = text
        if self.args.lower:
            input_text = input_text.lower()
        tokens, labels = self.get_prediction(input_text)
        return tokens, labels


class NERRuleExtractor(object):
    def __init__(self, regexes):
        self.regexes = regexes

    def extract(self, text):
        extracted_entities = []
        for entity_name, regex in self.regexes.items():
            entities = re.findall(regex, text)
            entities = [{entity_name: entity} for entity in entities]
            extracted_entities.extend(entities)
        return extracted_entities
