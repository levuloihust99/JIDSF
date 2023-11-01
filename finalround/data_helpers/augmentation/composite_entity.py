import re
import copy
import json
import torch
import random
import logging
import torch.nn.functional as F

from tqdm import tqdm
from utils.logging_utils import add_color_formatter
from transformers import BertTokenizer, BertForMaskedLM

from utils.utils import setup_random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)

from finalround.utils.vietnamese_words.trie import Trie

VIETNAMESE_WORDS_PATH = "finalround/utils/vietnamese_words/assets/words.txt"
vietnamese_words = []
with open(VIETNAMESE_WORDS_PATH, "r") as reader:
    for line in reader:
        line = line.strip()
        if line:
            vietnamese_words.append(line)

trie = Trie()
for word in vietnamese_words:
    trie.add(word)


RANDOM_SEED = 12345
VIETNAMESE_WORDS_PATH = "finalround/utils/vietnamese_words/assets/words.txt"
DATA_PATH = "final/data/ner/all.jsonl"
OUTPUT_PATH = "onboard/data/ner/entity_composite/all_composite_entity.jsonl"
TRACKING_FILE = "onboard/data/ner/entity_composite/tracker_entity_composite.json"


def load_data():
    data = []
    with open(DATA_PATH, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    return data


def get_entity(tagged_sequence):
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
                entities.append({
                    "entity_name": entity_name,
                    "idxs": entity_idxs,
                    "entity_tokens": [
                        tagged_sequence[_i][0]
                        for _i in entity_idxs
                    ]
                })
                entities[-1]["entity_value"] = " ".join(entities[-1]["entity_tokens"])
            else:
                while tagged_sequence[idx][1] == "I-{}".format(entity_name):
                    entity_idxs.append(idx)
                    idx += 1
                    if idx == len(tagged_sequence):
                        break
                entities.append({
                    "entity_name": entity_name,
                    "idxs": entity_idxs,
                    "entity_tokens": [
                        tagged_sequence[_i][0]
                        for _i in entity_idxs
                    ]
                })
                entities[-1]["entity_value"] = " ".join(entities[-1]["entity_tokens"])
    return entities


def get_all_idxs_sequence(*dims):
    if not dims:
        return [tuple()]
    follow_idxs_sequence = get_all_idxs_sequence(*dims[1:])
    all_idxs_sequence = []
    for i in range(dims[0]):
        for idxs in follow_idxs_sequence:
            all_idxs_sequence.append((i,) + idxs)
    return all_idxs_sequence


class CompositeEntityGenerator:
    """Add "của quân", "số 3",..."""

    def __init__(self):
        self.entity_numbers = list(range(1, 10))
        self.avail_owner = [
            "quân", "nam", "my", "trung", "hùng", "dũng", "lan", "thuỷ",
            "nguyên", "lâm", "trang", "linh", "sơn", "mạnh", "quyền", "quyết",
            "vượng", "ông bà", "bố mẹ", "long", "chiến", "tú", "vy", "duy anh",
            "bạn quân", "bạn nam", "bạn trường", "bố bạn linh", "mẹ bạn hải"
        ]
        self.avail_side = [
            "phải", "trái", "trên", "dưới", "kia", "này",
            "tả", "hữu", "đông", "tây", "nam", "bắc", "tây nam", "đông nam", "tây bắc", "đông bắc",
            "mạn phải", "mạn trái"
        ]
        self.tokenizer = BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
        self.model = BertForMaskedLM.from_pretrained("NlpHUST/vibert4news-base-cased")
        self.model.eval()

        vocab_mask = [False] * self.tokenizer.vocab_size
        for word in vietnamese_words:
            if word in self.tokenizer.vocab:
                vocab_mask[self.tokenizer.vocab[word]] = True
        vocab_mask = (1 - torch.tensor(vocab_mask).to(torch.long)).to(torch.bool)
        self.vocab_mask = vocab_mask
        self.exclude_words = []

    def augment(self, data):
        with open(TRACKING_FILE) as reader:
            tracker = json.load(reader)
        running_idx = tracker["running_idx"]
        existed_data = []
        with open(OUTPUT_PATH, "r") as reader:
            for line in reader:
                existed_data.append(json.loads(line.strip()))
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
        with open(OUTPUT_PATH, "w") as writer:
            for item in rewritten_existed_data:
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(OUTPUT_PATH, "a") as writer:
            for idx, item in enumerate(tqdm(data)):
                if idx < running_idx:
                    continue
                try:
                    tagged_sequence = list(zip(item["tokens"], item["labels"]))
                    self.tagged_sequence = tagged_sequence
                    augmentations = self.diversify()
                    for idx, augmentation in enumerate(augmentations):
                        augmentations[idx] = {
                            "file": item["file"],
                            **augmentation
                        }
                    for augmentation in augmentations:
                        writer.write(json.dumps(augmentation, ensure_ascii=False) + "\n")
                except KeyboardInterrupt as e:
                    with open(TRACKING_FILE, "w") as tracking_writer:
                        json.dump({"running_idx": idx}, tracking_writer)
                    logger.error(e)
                    exit(0)
                except Exception as e:
                    with open(TRACKING_FILE, "w") as tracking_writer:
                        json.dump({"running_idx": idx}, tracking_writer)
                    logger.error(e)
                    exit(0)

    def diversify(self):
        entities = get_entity(self.tagged_sequence)
        self.entities = entities
        entities_for_augment = []
        for entity in entities:
            if entity["entity_name"] not in {"device", "location", "scene"}:
                continue
            entity_value = entity["entity_value"]
            if "của" not in entity_value and "số" not in entity_value and "bên" not in entity_value:
                entities_for_augment.append(entity)
        
        if not entities_for_augment:
            return []
    
        variants = []
        for entity in entities_for_augment:
            variants.extend(self.entity_diversify(entity))
        return variants

    def entity_diversify(self, entity):
        variants = []
        if entity["entity_name"] == "device":
            variants.extend(self.add_owner(entity))
            variants.extend(self.add_quantity(entity))
            variants.extend(self.add_side(entity))
        elif entity["entity_name"] == "location":
            variants.extend(self.add_owner(entity))
            variants.extend(self.add_side(entity))
        elif entity["entity_name"] == "scene":
            variants.extend(self.scene_diversify(entity))
        return variants

    def add_owner(self, entity):
        assert entity["entity_name"] in {"device", "location"}
        flip_coin = random.random()
        fill_mask_template = "{entity} của {mask_token}".format(
            entity=entity["entity_value"],
            mask_token=self.tokenizer.mask_token
        )
        if flip_coin < 0.5:
            owner = self.lm_fill_mask(fill_mask_template, topk=3)[0]
        else:
            owner = random.choice(self.avail_owner)
        tagged_sequence = copy.deepcopy(self.tagged_sequence)
        idxs = entity["idxs"]
        entity_sequence = tagged_sequence[idxs[0] : idxs[-1] + 1]
        entity_sequence.extend([
            ("của", "I-{}".format(entity["entity_name"])),
            (owner, "I-{}".format(entity["entity_name"]))
        ])
        tagged_sequence = [
            *tagged_sequence[:idxs[0]],
            *entity_sequence,
            *tagged_sequence[idxs[-1] + 1:]
        ]
        return [
            {
                "tagged_sequence": tagged_sequence,
                "variant": "add_owner"
            }
        ]

    def add_quantity(self, entity):
        assert entity["entity_name"] == "device"
        entity_idxs = entity["idxs"]
        start_idx = entity_idxs[0]
        tagged_sequence = copy.deepcopy(self.tagged_sequence)
        if start_idx > 1:
            prev_token, prev_label = tagged_sequence[start_idx - 1]
            if prev_token in {"cái", "chiếc"}:
                prev_2_token, prev_2_label = tagged_sequence[start_idx - 2]
                match = re.match(r"^\d+$", prev_2_token)
                if not match:
                    added_quantity = random.choice(self.entity_numbers)
                    added_idx = start_idx - 1
                    tagged_sequence = [
                        *tagged_sequence[:added_idx],
                        ("{}".format(added_quantity), "O"),
                        *tagged_sequence[added_idx:]
                    ]
                    return [{
                        "tagged_sequence": tagged_sequence,
                        "variant": "add_quantity"
                    }]
        return []

    def add_side(self, entity):
        assert entity["entity_name"] in {"device", "location"}
        flip_coin = random.random()
        fill_mask_template = "{entity} bên {mask_token}".format(
            entity=entity["entity_value"],
            mask_token=self.tokenizer.mask_token
        )
        if flip_coin < 0.5:
            owner = self.lm_fill_mask(fill_mask_template, topk=3, exclude_words=[
                "trong", "ngoài", "cạnh", "mạn trong", "mạn ngoài"
            ])[0]
        else:
            owner = random.choice(self.avail_side)
        tagged_sequence = copy.deepcopy(self.tagged_sequence)
        idxs = entity["idxs"]
        entity_sequence = tagged_sequence[idxs[0] : idxs[-1] + 1]
        entity_sequence.extend([
            ("bên", "I-{}".format(entity["entity_name"])),
            (owner, "I-{}".format(entity["entity_name"]))
        ])
        tagged_sequence = [
            *tagged_sequence[:idxs[0]],
            *entity_sequence,
            *tagged_sequence[idxs[-1] + 1:]
        ]
        return [
            {
                "tagged_sequence": tagged_sequence,
                "variant": "add_owner"
            }
        ]

    def scene_diversify(self, entity):
        assert entity["entity_name"] == "scene"
        entity_value = entity["entity_value"]
        entity_idxs = entity["idxs"]
        cand_idxs = []

        idx = 0
        while idx < len(entity_idxs):
            token = entity["entity_tokens"][idx]
            if token == "chế":
                if idx < len(entity_idxs) - 1 and entity["entity_tokens"][idx + 1] == "độ":
                    idx += 2
                else:
                    cand_idxs.append(entity_idxs[idx])
                    idx += 1
            else:
                cand_idxs.append(entity_idxs[idx])
                idx += 1

        if not cand_idxs:
            return []

        selected_idx_for_mask = random.choice(cand_idxs)
        tagged_sequence = copy.deepcopy(self.tagged_sequence)
        tokens = [item[0] for item in tagged_sequence]
        masked_word = tokens[selected_idx_for_mask]
        tokens[selected_idx_for_mask] = self.tokenizer.mask_token
        fill_mask_template = " ".join(tokens)
        pred_word = self.lm_fill_mask(fill_mask_template, topk=1)[0]
        if pred_word == masked_word:
            return []
        tokens[selected_idx_for_mask] = pred_word
        labels = [item[1] for item in tagged_sequence]
        return [
            {
                "tagged_sequence": list(zip(tokens, labels)),
                "variant": "scene_diversify"
            }
        ]

    def lm_fill_mask(self, text, topk, exclude_words=None):
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids])

        with torch.no_grad():
            outputs = self.model(input_ids, return_dict=True)
        active_mask = input_ids.squeeze().eq(self.tokenizer.mask_token_id) # [seq_len]
        logits = outputs.logits.squeeze(dim=0)[active_mask] # [num_mask, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1) # [num_mask, vocab_size]

        exclude_mask = None
        this_exclude_words = exclude_words or self.exclude_words
        if this_exclude_words:
            exclude_mask = []
            for word in this_exclude_words:
                if word in self.tokenizer.vocab:
                    exclude_mask.append(self.tokenizer.vocab[word])
            exclude_mask = torch.tensor(exclude_mask)
        
        this_vocab_mask = self.vocab_mask.clone()
        if exclude_mask is not None:
            this_vocab_mask[exclude_mask] = True
        log_probs[:, this_vocab_mask] = -1e20

        all_mask_scores, all_mask_pred_cands = torch.topk(log_probs, topk, dim=-1) # [num_mask, topk]
        idxs_sequence = get_all_idxs_sequence(*([topk] * all_mask_scores.size(0)))
        cand_sequence = []
        for seq in idxs_sequence:
            seq_score = []
            seq_token_ids = []
            reduced_seq_score = 0.0
            for i, j in enumerate(seq):
                token_score = all_mask_scores[i][j].item()
                reduced_seq_score += token_score
                seq_score.append(token_score)
                seq_token_ids.append(all_mask_pred_cands[i][j].item())
            cand_sequence.append({
                "seq_score": seq_score,
                "reduced_seq_score": reduced_seq_score,
                "seq_token_ids": seq_token_ids,
            })

        cand_sequence = sorted(cand_sequence, key=lambda x: x["reduced_seq_score"], reverse=True)
        topk_cand_sequence = cand_sequence[:topk]
        selected_cand_sequence = random.choice(topk_cand_sequence)
        seq_token_ids = selected_cand_sequence["seq_token_ids"]
        seq_tokens = self.tokenizer.convert_ids_to_tokens(seq_token_ids)
        return seq_tokens

def main():
    augmenter = CompositeEntityGenerator()
    data = load_data()
    setup_random(RANDOM_SEED)
    augmenter.augment(data)


if __name__ == "__main__":
    main()
