import os
import time
import copy
import json
import torch
import random
import logging
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from typing import Text, List, Dict, Any
from transformers import BertForMaskedLM, RobertaForMaskedLM, BertTokenizer, PhobertTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

from utils.logging_utils import add_color_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)

basic_tokenizer = BasicTokenizer(
    do_lower_case=True,
    strip_accents=False,
    do_split_on_punc=True
)

from finalround.utils.vietnamese_words.trie import Trie
from utils.utils import setup_random

VIETNAMESE_WORDS_PATH = "finalround/utils/vietnamese_words/assets/words.txt"


def load_data(data_path):
    data = []
    with open(data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    return data


def load_bert4news():
    tokenizer = BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
    model = BertForMaskedLM.from_pretrained("NlpHUST/vibert4news-base-cased")
    model.eval()
    return tokenizer, model


def load_phobert():
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
    model = RobertaForMaskedLM.from_pretrained("vinai/phobert-base")
    model.eval()
    return tokenizer, model


def get_all_idxs_sequence(*dims):
    if not dims:
        return [tuple()]
    follow_idxs_sequence = get_all_idxs_sequence(*dims[1:])
    all_idxs_sequence = []
    for i in range(dims[0]):
        for idxs in follow_idxs_sequence:
            all_idxs_sequence.append((i,) + idxs)
    return all_idxs_sequence


class Diversifier:
    def __init__(self, trie: Trie, vietnamese_words, type: Text = "bert4news"):
        if type == "bert4news":
            tokenizer, model = load_bert4news()
        else:
            tokenizer, model = load_phobert()
        self.vietnamese_words = vietnamese_words
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.trie = trie
        self.avail_words = [
            word for word in self.tokenizer.vocab if self.trie.exists(word)
        ]
        self.exclude_words = [
            "tăng", "giảm", "kiểm", "tra", "bật", "mở", "đóng", "tắt",
            "tiệc", "tắm", "phòng", "sân", "hiên", "vườn", "thang",
            "riêng", "tư", "sảnh", "giặt", "bếp", "đèn", "giãn",
            "nóng", "lạnh", "nâng", "check", "chạy", "ấm", "mát",
            "cửa", "giường", "nhà", "sưởi", "quạt"
        ]
        self.avail_words_exclusive = [
            word for word in self.avail_words if word not in self.exclude_words
        ]
        vocab_mask = [False] * tokenizer.vocab_size
        for word in vietnamese_words:
            if word in tokenizer.vocab:
                vocab_mask[tokenizer.vocab[word]] = True
        vocab_mask = (1 - torch.tensor(vocab_mask).to(torch.long)).to(torch.bool)
        self.vocab_mask = vocab_mask

    def diversify(self, data, tracking_file, output_path):
        # setup tracking file
        if os.path.exists(tracking_file):
            with open(tracking_file) as reader:
                tracker = json.load(reader)
        else:
            tracking_file_dir = os.path.dirname(tracking_file)
            if not os.path.exists(tracking_file_dir):
                os.makedirs(tracking_file_dir)
            tracker = {}
        running_idx = tracker.get("running_idx", 0)

        # check for existed data
        existed_data = []
        if os.path.exists(output_path):
            with open(output_path, "r") as reader:
                for line in reader:
                    existed_data.append(json.loads(line.strip()))
        else:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            existed_data = []

        # rewrite existed file for consistency
        file_id_tracker = set()
        existed_file_ids = []
        for item in existed_data:
            if item["file"] not in file_id_tracker:
                file_id_tracker.add(item["file"])
                existed_file_ids.append(item["file"])
        existed_file_ids = existed_file_ids[:running_idx]
        existed_file_id_tracker = set(existed_file_ids)
        rewritten_existed_data = []
        for item in existed_data:
            if item["file"] in existed_file_id_tracker:
                rewritten_existed_data.append(item)
        with open(output_path, "w") as writer:
            for item in rewritten_existed_data:
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")

        # loop for creating augmentations
        with open(output_path, "a") as writer:
            for idx, item in enumerate(tqdm(data)):
                if idx < running_idx:
                    continue
                try:
                    tagged_sequence = list(zip(item["tokens"], item["labels"]))
                    variants = self.text_diversify(tagged_sequence)
                    for _idx, variant in enumerate(variants):
                        variants[_idx] = {"file": item["file"], **variant}
                    for variant in variants:
                        writer.write(json.dumps(variant, ensure_ascii=False) + "\n")
                except KeyboardInterrupt as e:
                    with open(tracking_file, "w") as tracking_writer:
                        json.dump({"running_idx": idx}, tracking_writer)
                    logger.error(e)
                    exit(0)
                except Exception as e:
                    with open(tracking_file, "w") as tracking_writer:
                        json.dump({"running_idx": idx}, tracking_writer)
                    logger.error(e)
                    exit(0)
        
        with open(tracking_file, "w") as writer:
            json.dump({"running_idx": idx + 1}, writer)

    def text_diversify(self, tagged_sequence):
        added_token_variant = self.add_lm_token(tagged_sequence)
        replaced_token_variant = self.replace_lm_token(tagged_sequence)
        replaced_entity_variant = self.replace_entity(tagged_sequence)
        swap_variant = self.token_swap(tagged_sequence)
        noise_word_variant = self.add_noise_word(tagged_sequence)
        replace_by_noise_variant = self.replace_by_noise_word(tagged_sequence)
        augmented_sequences = [
            {
                "tagged_sequence": tagged_sequence,
                "variant": "origin"
            },
            *added_token_variant,
            *replaced_token_variant,
            *replaced_entity_variant,
            *swap_variant,
            *noise_word_variant,
            *replace_by_noise_variant
        ]
        return augmented_sequences

    def replace_mask_token(
        self,
        augmented_tokens,
        masked_words = None,
        topk: int = 10,
        force_replace: bool = False,
        should_exclude: bool = False
    ):
        """`augmented_tokens` have mask tokens"""

        augmented_tokens = copy.deepcopy(augmented_tokens)
        subword_tokens = []
        for token in augmented_tokens:
            subword_tokens.extend(self.tokenizer.tokenize(token))
        subword_tokens = [self.tokenizer.cls_token] + subword_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        input_ids = torch.tensor([input_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, return_dict=True)
        logits = outputs.logits.squeeze() # [seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        active_mask = input_ids.squeeze().eq(self.tokenizer.mask_token_id) # [seq_len]

        if force_replace:
            mask_idxs = active_mask.nonzero().squeeze(dim=-1).tolist()
            mask_idxs = [(i, idx) for i, idx in enumerate(mask_idxs)]
            i, selected_mask_idx = random.choice(mask_idxs)
            try:
                masked_id = self.tokenizer.convert_tokens_to_ids([masked_words[i]])[0]
                log_probs[selected_mask_idx, masked_id] = -1e20
            except Exception as e:
                logger.error(e)

        exclude_mask = None
        if should_exclude:
            exclude_mask = []
            for word in self.exclude_words:
                if word in self.tokenizer.vocab:
                    exclude_mask.append(self.tokenizer.vocab[word])
            if exclude_mask:
                exclude_mask = torch.tensor(exclude_mask)
            else:
                exclude_mask = None

        this_vocab_mask = self.vocab_mask.clone()
        if exclude_mask is not None:
            this_vocab_mask[exclude_mask] = True
        log_probs[:, this_vocab_mask] = -1e20

        scores, vocab_indices = torch.topk(log_probs, topk, dim=-1) # [seq_len, topk]
        all_mask_pred_cands = vocab_indices[active_mask] # [num_mask, topk]
        all_mask_scores = scores[active_mask] # [num_mask, topk]
        idxs_sequence = get_all_idxs_sequence(*([topk] * all_mask_scores.size(0)))
        cand_sequence = []
        for seq in idxs_sequence:
            seq_score = 0.0
            seq_token_ids = []
            for i, j in enumerate(seq):
                seq_score += all_mask_scores[i][j].item()
                seq_token_ids.append(all_mask_pred_cands[i][j].item())
            cand_sequence.append((seq_score, seq_token_ids))
        cand_sequence = sorted(cand_sequence, key=lambda x: x[0], reverse=True)
        topk_pred_ids = cand_sequence[:topk]
        topk_pred_ids = [pred[1] for pred in topk_pred_ids]
        topk_pred_words = [
            self.tokenizer.convert_ids_to_tokens(pred_ids)
            for pred_ids in topk_pred_ids
        ]
        selected_words = random.choice(topk_pred_words)

        mask_idx = 0
        ignored_idxs = []
        for idx, word in enumerate(augmented_tokens):
            if word == self.tokenizer.mask_token:
                if selected_words[mask_idx] is not None:
                    augmented_tokens[idx] = selected_words[mask_idx]
                    mask_idx += 1
                    if mask_idx == len(selected_words):
                        break
                else:
                    ignored_idxs.append(idx)
                    augmented_tokens[idx] = None
        augmented_tokens = [token for token in augmented_tokens if token is not None]
        return augmented_tokens, ignored_idxs

    def find_candidate_idxs_for_add(self, tagged_sequence):
        candidate_idxs = []
        for idx, (token, label) in enumerate(tagged_sequence):
            if (
                (idx == 0 and label == "O") or
                (idx > 0 and label == "O" and tagged_sequence[idx - 1][1] == "O")
            ):
                candidate_idxs.append(idx)
        return candidate_idxs

    def add_lm_token(self, tagged_sequence):
        """Only add between two O tokens."""

        candidate_idxs = self.find_candidate_idxs_for_add(tagged_sequence)
        if not candidate_idxs:
            return []

        selected_idx = random.choice(candidate_idxs)
        augmented_tokens = []
        augmented_labels = []

        idx = 0
        while idx < len(tagged_sequence):
            if idx == selected_idx:
                augmented_tokens.extend([self.tokenizer.mask_token, tagged_sequence[idx][0]])
                augmented_labels.extend(["O", tagged_sequence[idx][1]])
            else:
                augmented_tokens.append(tagged_sequence[idx][0])
                augmented_labels.append(tagged_sequence[idx][1])
            idx += 1
        
        augmented_tokens, ignored_idxs = self.replace_mask_token(
            augmented_tokens, topk=3, should_exclude=True)
        augmented_labels = [
            label for idx, label in enumerate(augmented_labels) if idx not in ignored_idxs
        ]
        return [{
            "tagged_sequence": list(zip(augmented_tokens, augmented_labels)),
            "variant": "add_lm_token"
        }]

    def find_candidate_idxs_for_replace(self, tagged_sequence):
        candidate_idxs = []
        for idx, (token, label) in enumerate(tagged_sequence):
            should_be_candidate = False
            if (
                idx == 0 and
                len(tagged_sequence) > 0 and
                label == "O" and
                tagged_sequence[idx + 1][1] == "O"
            ):
                should_be_candidate = True
            elif (
                idx == len(tagged_sequence) - 1 and
                label == "O" and
                tagged_sequence[idx - 1][1] == "O"
            ):
                should_be_candidate = True
            elif (
                label == "O"
                and tagged_sequence[idx - 1][1] == "O"
                and tagged_sequence[idx + 1][1] == "O"
            ):
                should_be_candidate = True
            if should_be_candidate is True:
                candidate_idxs.append(idx)
        return candidate_idxs

    def replace_lm_token(self, tagged_sequence):
        """Only replace token that sit between two O tokens."""

        candidate_idxs = self.find_candidate_idxs_for_replace(tagged_sequence)
        if not candidate_idxs:
            return []

        selected_idx = random.choice(candidate_idxs)
        augmented_tokens = []
        augmented_labels = []
        masked_words = []
        for idx, (token, label) in enumerate(tagged_sequence):
            if idx == selected_idx:
                masked_words.append(token)
                augmented_tokens.append(self.tokenizer.mask_token)
                augmented_labels.append("O")
            else:
                augmented_tokens.append(token)
                augmented_labels.append(label)
        
        augmented_tokens, ignored_idxs = self.replace_mask_token(
            augmented_tokens, topk=3, masked_words=masked_words, force_replace=True, should_exclude=True)
        augmented_labels = [
            label for idx, label in enumerate(augmented_labels) if idx not in ignored_idxs
        ]
        return [{
            "tagged_sequence": list(zip(augmented_tokens, augmented_labels)),
            "variant": "replace_lm_token"
        }]

    def do_entity_replace(self, tagged_sequence, entity_idxs):
        tagged_sequence = copy.deepcopy(tagged_sequence)
        max_num_mask = min(len(entity_idxs), 1)
        ns_mask = list(range(1, max_num_mask + 1))
        n_mask = random.choice(ns_mask)
        mask_idxs = random.sample(entity_idxs, n_mask)
        augmented_tokens = [token for token, label in tagged_sequence]
        masked_words = []
        for idx in mask_idxs:
            masked_words.append(augmented_tokens[idx])
            augmented_tokens[idx] = self.tokenizer.mask_token
        augmented_tokens, ignored_idxs = self.replace_mask_token(
            augmented_tokens, topk=3, masked_words=masked_words, force_replace=True)
        if len(augmented_tokens) == 0:
            return []

        augmented_labels = [
            label for idx, (token, label) in enumerate(tagged_sequence) if idx not in ignored_idxs
        ]
        return [{
            "tagged_sequence": list(zip(augmented_tokens, augmented_labels)),
            "variant": "replace_entity"
        }]

    def replace_entity(self, tagged_sequence):
        entities = []
        idx = 0
        while idx < len(tagged_sequence):
            token, label = tagged_sequence[idx]
            assert label.startswith("I-") is False
            if label == "O":
                idx += 1
            else: # starts with "B-"
                entity_name = label[2:]
                entity_idxs = [idx]
                idx += 1
                if idx == len(tagged_sequence):
                    entities.append({"entity_name": entity_name, "idxs": entity_idxs})
                else:
                    while tagged_sequence[idx][1] == "I-{}".format(entity_name):
                        entity_idxs.append(idx)
                        idx += 1
                        if idx == len(tagged_sequence):
                            break
                    entities.append({"entity_name": entity_name, "idxs": entity_idxs})

        entities = [
            entity for entity in entities
            if entity["entity_name"] in {"scene", "device", "location"}
        ]
        if not entities:
            return []

        augmentations = []
        for selected_entity in entities:
            augmentations.extend(self.do_entity_replace(tagged_sequence, selected_entity["idxs"]))
        return augmentations

    def token_swap(self, tagged_sequence):
        """Swap two O tokens that sit among an array of O tokens."""

        tagged_sequence = copy.deepcopy(tagged_sequence)
        O_arrays = []
        idx = 0
        while idx < len(tagged_sequence):
            token, label = tagged_sequence[idx]
            if label == "O":
                O_idxs = []
                while label == "O":
                    O_idxs.append(idx)
                    idx += 1
                    if idx == len(tagged_sequence):
                        break
                    token, label = tagged_sequence[idx]
                O_arrays.append(O_idxs)
            idx += 1
        
        cands_O_idxs = []
        for O_idxs in O_arrays:
            start_idx = O_idxs[0]
            if start_idx != 0:
                this_start_idx = 1
            else:
                this_start_idx = 0
            end_idx = O_idxs[-1]
            if end_idx != len(tagged_sequence) - 1:
                this_end_idx = -1
            else:
                this_end_idx = len(O_idxs)
            cands_O_idxs.append(O_idxs[this_start_idx : this_end_idx])
        cands_O_idxs = [O_idxs for O_idxs in cands_O_idxs if len(O_idxs) > 1]

        if not cands_O_idxs:
            return []

        selected_O_idxs = random.choice(cands_O_idxs)
        idx1, idx2 = random.sample(selected_O_idxs, 2)

        # swap
        tmp = tagged_sequence[idx1]
        tagged_sequence[idx1] = tagged_sequence[idx2]
        tagged_sequence[idx2] = tmp

        return [{
            "tagged_sequence": tagged_sequence,
            "variant": "token_swap"
        }]

    def add_noise_word(self, tagged_sequence):
        """Only add between two O tokens."""

        candidate_idxs = self.find_candidate_idxs_for_add(tagged_sequence)
        if not candidate_idxs:
            return []

        tagged_sequence = copy.deepcopy(tagged_sequence)
        selected_idx = random.choice(candidate_idxs)

        augmented_tokens = []
        augmented_labels = []

        noise_word = random.choice(self.avail_words_exclusive)
        idx = 0
        while idx < len(tagged_sequence):
            if idx == selected_idx:
                augmented_tokens.extend([noise_word, tagged_sequence[idx][0]])
                augmented_labels.extend(["O", tagged_sequence[idx][1]])
            else:
                augmented_tokens.append(tagged_sequence[idx][0])
                augmented_labels.append(tagged_sequence[idx][1])
            idx += 1

        return [{
            "tagged_sequence": list(zip(augmented_tokens, augmented_labels)),
            "variant": "add_noise_word"
        }]

    def replace_by_noise_word(self, tagged_sequence):
        """Only replace token that sit between two O tokens."""

        candidate_idxs = self.find_candidate_idxs_for_replace(tagged_sequence)
        if not candidate_idxs:
            return []

        tagged_sequence = copy.deepcopy(tagged_sequence)
        selected_idx = random.choice(candidate_idxs)
        word_for_replace = tagged_sequence[selected_idx][0]
        num_try = 0
        max_tries = 10
        while num_try < max_tries:
            noise_word = random.choice(self.avail_words_exclusive)
            if noise_word != word_for_replace:
                tagged_sequence[selected_idx] = (noise_word, tagged_sequence[selected_idx][1])
                break
            num_try += 1
        if num_try == max_tries:
            return []

        return [{
            "tagged_sequence": tagged_sequence,
            "variant": "replace_by_noise_word"
        }]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/ner/all.jsonl")
    parser.add_argument("--tracking_file", default="data/ner/augmented/lm/tracking.json")
    parser.add_argument("--output_path", default="data/ner/augmented/lm/lm_augmented.jsonl")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    vietnamese_words = []
    with open(VIETNAMESE_WORDS_PATH, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                vietnamese_words.append(line)
    
    trie = Trie()
    for word in vietnamese_words:
        trie.add(word)

    setup_random(args.seed)
    diversifier = Diversifier(trie, vietnamese_words, type="bert4news")

    data = load_data(args.data_path)
    diversifier.diversify(data, tracking_file=args.tracking_file, output_path=args.output_path)


if __name__ == "__main__":
    main()
