import re
import torch
import requests
import logging

import copy
import unicodedata
import unidecode
import numpy as np
from typing import Literal
from transformers import BertTokenizer
basic_tokenizer = BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased").basic_tokenizer

logger = logging.getLogger(__name__)

def extract_entities(
    text,
    tokens,
    labels,
    architecture: Literal["bert", "roberta"]
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
            if prev_token_is_entity:
                current_idxs.append(idx)
            if idx == len(labels) - 1 and prev_token_is_entity:
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

        idx += 1
        if idx == len(labels):
            break

    tokens = decode_subword_token(tokens, architecture)
    space_ocurrences = spaces_determine(text)

    L = len(text)
    N = len(space_ocurrences)

    positions, count = zip(*space_ocurrences)
    count_accum = [sum(count[:idx]) for idx in range(1, N + 1)]
    count_accum_pseudo = [0] + count_accum
    boundaries = [0] + [positions[idx] - count_accum_pseudo[idx] for idx in range(N)] + [L]
    intervals = [(boundaries[idx], boundaries[idx + 1]) for idx in range(0, N + 1)]

    num_tokens = len(tokens)
    token_lengths = [len(token) for token in tokens]
    token_occurrences = [sum(token_lengths[:idx]) for idx in range(num_tokens)]

    cache = {
        "text": text,
        "intervals": intervals,
        "token_lengths": token_lengths,
        "token_occurences": token_occurrences,
        "count_accum_pseudo": count_accum_pseudo
    }
    entities = [align(entity, cache) for entity in entities]
    return entities


def align(entity, cache):
    text = cache.get("text")
    token_lengths = cache.get("token_lengths")
    token_occurrences = cache.get("token_occurences")
    count_accum_pseudo = cache.get("count_accum_pseudo")
    intervals = cache.get("intervals")

    token_indexes = entity["indexes"]
    entity_token_occurences = [token_occurrences[idx] for idx in token_indexes]
    entity_token_lengths = [token_lengths[idx] for idx in token_indexes]

    try:
        entity_start = entity_token_occurences[0]
    except Exception:
        pass
    entity_start_resolved = (
        entity_start + count_accum_pseudo[binary_search(intervals, entity_start)]
    )
    entity_end = entity_token_occurences[-1] + entity_token_lengths[-1] - 1
    entity_end_resolved = (
        entity_end + count_accum_pseudo[binary_search(intervals, entity_end)] + 1
    )
    entity = {
        "entity": entity["entity"],
        "start": entity_start_resolved,
        "end": entity_end_resolved,
        "value": text[entity_start_resolved : entity_end_resolved],
        "indexes": token_indexes
    }
    return entity


def binary_search(intervals, value):
    start = 0
    end = len(intervals)
    while start <= end:
        mid = (start + end) // 2
        if intervals[mid][0] <= value < intervals[mid][1]:
            return mid
        if value < intervals[mid][0]:
            end = mid - 1
        elif value >= intervals[mid][1]:
            start = mid + 1
    return -1


def decode_subword_token(tokens, architecture):
    if architecture == "bert":
        tokens = [token[2:] if token.startswith("##") else token for token in tokens]
    elif architecture == "roberta":
        tokens = [token[:-2] if token.endswith("@@") else token for token in tokens]
    return tokens


def spaces_determine(text):
    space_occurences = []
    is_previous_space = False
    idx = 0
    while idx < len(text):
        if text[idx] == " ":
            if not is_previous_space:
                is_previous_space = True
                pos = idx
                count = 1
            else:
                count += 1
                if idx == len(text) - 1:
                    space_occurences.append((pos, count))
        else:
            if is_previous_space:
                space_occurences.append((pos, count))
            is_previous_space = False
        idx += 1
    if not space_occurences:
        space_occurences = [(len(text), 0)]
    return space_occurences


def align_segmented_sentence(sentence, segmented_sentence):
    tokens = sentence.strip().split()
    origin_tokens = copy.deepcopy(tokens)
    segmented_tokens = segmented_sentence.strip().split()

    out_segmented_tokens = []
    marked = 0  # track the current position in `tokens` variable
    for idx, segmented_token in enumerate(segmented_tokens):  # inner_idx for debug
        aligned_token = tokens[marked]

        # one-to-multi case, i.e. one origin token becomes multiple segmented tokens
        #    E.g. origin sentence: "(Đợi hoài)"
        #    non-segmented tokens: ["(Đợi", "hoài)"]
        #        segmented tokens: ["(", "Đợi", "hoài_)"]
        #           aligned token: "(Đợi"
        #         segmented token: "("
        if len(segmented_token) < len(aligned_token):
            front_token = aligned_token[:len(segmented_token)]
            back_token = aligned_token[len(segmented_token):]
            tokens = tokens[:marked] + [front_token, back_token] + tokens[marked + 1:]
            out_segmented_tokens += [segmented_token]
            marked += 1
        elif len(segmented_token) == len(aligned_token):
            out_segmented_tokens += [segmented_token]
            if idx < len(segmented_tokens) - 1:
                out_segmented_tokens += [" "]
            marked += 1
        else:  # multi-to-one case, i.e. several origin tokens are combined into one segmented token
            atomic_tokens = segmented_token.split("_")  # E.g. segmented_token = 'Lê_Văn_Lương' --> atomic_tokens = ['Lê', 'Văn', 'Lương']
            for i in range(len(atomic_tokens)):
                aligned_token = tokens[marked]
                # 1. the atomic token equals the aligned token
                if len(atomic_tokens[i]) == len(aligned_token):
                    out_segmented_tokens += [atomic_tokens[i]]
                    if i < len(atomic_tokens) - 1:
                        out_segmented_tokens += ["_"]
                    elif idx < len(segmented_tokens) - 1:
                        out_segmented_tokens += [" "]
                # 3. E.g. origin sentence: "Ngày hôm nay, tính đến 6h00, Hà Nội ghi nhận 15 ca mắc mới"
                #    non-segmented tokens: ["Ngày", "hôm", "nay,",...]
                #        segmented tokens: ["Ngày", "hôm_nay", ",", ...]
                #           atomic_tokens: ["hôm", "nay"]
                #           aligned_token: "nay,"
                elif len(aligned_token) > len(atomic_tokens[i]):
                    # split the aligned token, i.e. "nay," --> ["nay", ","]
                    front_token = aligned_token[:len(atomic_tokens[i])]
                    back_token = aligned_token[len(atomic_tokens[i]):]
                    # change the origin tokens, i.e. alternate the aligned token by the splitted tokens
                    tokens = tokens[:marked] + [front_token, back_token] + tokens[marked + 1:]
                    out_segmented_tokens += [atomic_tokens[i]]
                else:
                    raise ValueError(
                        "Encounter error when aligning two sentences:\n\t{}\n\t{}\nCurrent atomic token = {}\nSegmented token = {}\nAligned token = {}".format(
                            sentence, segmented_sentence, atomic_tokens[i], segmented_token, aligned_token
                        )
                    )
                marked += 1

    assert len(" ".join(origin_tokens)) == len("".join(out_segmented_tokens))
    return "".join(out_segmented_tokens)


class NERProcessor(object):
    def __init__(
        self,
        ner_tokenizer,
        ner_model,
        ner_label_mappings,
        segmenter,
        args
    ):
        self.ner_tokenizer = ner_tokenizer
        self.ner_model = ner_model
        self.ner_label_mappings = ner_label_mappings
        self.inverse_ner_label_mappings = {v : k for k, v in ner_label_mappings.items()}
        self.segmenter = segmenter
        self.args = args

        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(args.gpu_id))
            logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
            logger.info('We will use the GPU:{}, {}'.format(torch.cuda.get_device_name(args.gpu_id), torch.cuda.get_device_capability(config.gpu_id)))
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     logger.info("MPS backend is available, using MPS.")
        else:
            logger.info('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        
        self.ner_model.to(self.device)
        self.ner_model.eval()
    
    def segment(self, text):
        return self.segmenter.segment(text)

    def get_prediction(self, text):
        # words = text.split()
        # words_wo_puncs = []
        # for word in words:
        #     if re.search(r"[.!?,:;\[\]\(\)]", word):
        #         words_wo_puncs.extend(basic_tokenizer.tokenize(word))
        #     else:
        #         words_wo_puncs.append(word)

        # tokens = []
        # for word in words_wo_puncs:
        #     tokens.extend(self.ner_tokenizer.tokenize(word))
        tokens = self.ner_tokenizer.tokenize(text)
        tokens = [self.ner_tokenizer.cls_token] + tokens + [self.ner_tokenizer.sep_token]
        token_ids = self.ner_tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.ner_model.forward(input_ids=input_ids)

        sequence_output = outputs[0]
        sequence_output = torch.squeeze(sequence_output, dim=0)
        
        predict_labels = torch.argmax(sequence_output, dim=-1)
        predict_labels = predict_labels.cpu().numpy().tolist()
        predict_labels = [self.inverse_ner_label_mappings[label_id] for label_id in predict_labels]

        return tokens[1:-1], predict_labels[1:-1] # ignore special tokens

    @staticmethod
    def extract_entities(
        tokens,
        labels,
    ):
        assert len(tokens) == len(labels)
        entities = []
        idx = 0
        current_label = None
        current_entity = []
        prev_token_is_entity = False
        
        while True:
            label = labels[idx]
            if label.startswith('B-'):
                current_label = label[2:]
                current_entity.append(tokens[idx])
                prev_token_is_entity = True
            elif label.startswith('I-'):
                if prev_token_is_entity:
                    current_entity.append(tokens[idx])
            else:
                if prev_token_is_entity:
                    entities.append((' '.join(current_entity), current_label))
                    current_entity = []
                    current_label = None
                    prev_token_is_entity = False
            idx += 1
            if idx == len(labels):
                break
        
        entities = [{k : v} for v, k in entities]
        return entities

    def extract(self, text):
        architecture = "bert"
        input_text = text
        if self.segmenter:
            input_text = self.segment(text)
            # input_text = align_segmented_sentence(text, input_text)
            architecture = "roberta"
        tokens, labels = self.get_prediction(input_text)
        entities = extract_entities(input_text, tokens, labels, architecture)
        for entity in entities:
            entity["value"] = input_text[entity["start"] : entity["end"]]
            if self.segmenter:
                entity["value"] = entity["value"].replace("_", " ")
            entity.pop("indexes")
        return entities


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
