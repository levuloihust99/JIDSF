import io
import copy
import torch
import logging
import torch.nn.functional as F

from nltk import sent_tokenize
from typing import Literal, List, Dict, Text, Any, Optional
from finalround.modeling.ner_cont import BertNERCont
from transformers.models.bert.tokenization_bert import BasicTokenizer

from text_tools.text_preprocess.preprocessing import _is_whitespace

basic_tokenizer = BasicTokenizer(
    do_lower_case=False,
    strip_accents=False,
    do_split_on_punc=True
)

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
    space_ocurrences = whitespaces_determine(text)

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


def whitespaces_determine(text):
    space_occurences = []
    is_previous_space = False
    idx = 0
    while idx < len(text):
        if _is_whitespace(text[idx]):
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

    # def get_prediction(self, text: List[Text], tracker: Optional[Dict] = None):
    #     tokens = self.tokenizer.tokenize(text)
    #     tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
    #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

    #     input_ids = torch.tensor([token_ids], dtype=torch.long)
    #     input_ids = input_ids.to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model.forward(input_ids=input_ids)

    #     sequence_output = outputs[0]
    #     sequence_output = torch.squeeze(sequence_output, dim=0) # [seq_len, num_labels]

    #     if isinstance(tracker, dict):
    #         log_probs = F.log_softmax(sequence_output, dim=-1)
    #         log_probs = torch.max(log_probs, dim=-1).values # [seq_len]
    #         tracker["log_probs"] = log_probs[1:-1]
        
    #     predict_labels = torch.argmax(sequence_output, dim=-1)
    #     predict_labels = predict_labels.cpu().numpy().tolist()
    #     predict_labels = [self.inverse_label_mappings[label_id] for label_id in predict_labels]

    #     return tokens[1:-1], predict_labels[1:-1] # ignore special tokens

    def get_prediction(self, texts: List[Text], tracker: Optional[Dict] = None):
        inputs = self.tokenizer(texts, max_length=self.model.config.max_position_embeddings, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0] # [bsz, seq_len, num_labels]
        num_labels = sequence_output.size(-1)
        flatten_input_ids = input_ids.view(-1)
        flatten_attention_mask = attention_mask.view(-1)
        active_mask = (
            (flatten_attention_mask == 1) &
            (flatten_input_ids != self.tokenizer.cls_token_id) &
            (flatten_input_ids != self.tokenizer.sep_token_id)
        )
        active_sequence_output = sequence_output.view(-1, num_labels)[active_mask] # [ext_seq_len, num_labels]

        if isinstance(tracker, dict):
            log_probs = F.log_softmax(active_sequence_output, dim=-1)
            log_probs = torch.max(log_probs, dim=-1).values # [ext_seq_len]
            tracker["log_probs"] = log_probs

        predict_labels = torch.argmax(active_sequence_output, dim=-1)
        predict_labels = predict_labels.cpu().numpy().tolist()
        predict_labels = [self.inverse_label_mappings[label_id] for label_id in predict_labels]

        active_input_ids = flatten_input_ids[active_mask]
        active_input_tokens = self.tokenizer.convert_ids_to_tokens(active_input_ids)
        return active_input_tokens, predict_labels

    def pre_extract(self, text):
        input_text = text
        if self.segmenter:
            input_text = self.segment(input_text)
        punc_separated_text = " ".join(basic_tokenizer.tokenize(input_text))
        if self.args.lower:
            punc_separated_text = punc_separated_text.lower()
        return punc_separated_text, input_text

    # def extract(self, text):
    #     punc_separated_text, architecture, input_text = self.pre_extract(text)
    #     tracker = {}
    #     tokens, labels = self.get_prediction(punc_separated_text, tracker=tracker)
    #     log_probs = tracker["log_probs"]
    #     entities = extract_entities(input_text, tokens, labels, architecture)
    #     for entity in entities:
    #         indexes = entity.pop("indexes")
    #         confidence_entity = torch.exp(torch.mean(log_probs[torch.tensor(indexes)])).item()
    #         entity["confidence_entity"] = confidence_entity
    #     entities = self.post_extract(input_text, entities)
    #     return entities

    def post_extract(self, entities: List[Dict[Text, Any]]):
        entities = copy.deepcopy(entities)
        for entity in entities:
            if self.segmenter:
                entity["value"] = entity["value"].replace("_", " ")
        return entities

    def split_text(self, text: Text):
        stream = io.StringIO(text)
        sentences = []
        for line in stream:
            line = line.strip()
            if line:
                sents = sent_tokenize(line)
                sentences.extend(sents)
        return sentences

    def extract(self, text: Text):
        tracker = {}
        if self.args.do_split:
            chunks = self.split_text(text)
            punc_separated_chunks = []
            input_chunks = []
            for chunk in chunks:
                punc_separated_chunk, input_chunk = self.pre_extract(chunk)
                punc_separated_chunks.append(punc_separated_chunk)
                input_chunks.append(input_chunk)
            tokens, labels = self.get_prediction(punc_separated_chunks, tracker=tracker)
        else:
            punc_separated_text, input_text = self.pre_extract(text)
            tokens, labels = self.get_prediction([punc_separated_text], tracker=tracker)

        log_probs = tracker["log_probs"]
        architecture = "bert" if "bert" in self.args.model_type else "roberta"
        entities = extract_entities(text, tokens, labels, architecture)
        for entity in entities:
            indexes = entity.pop("indexes")
            confidence_entity = torch.exp(torch.mean(log_probs[torch.tensor(indexes)])).item()
            entity["confidence_entity"] = confidence_entity
        entities = self.post_extract(entities)
        return entities
