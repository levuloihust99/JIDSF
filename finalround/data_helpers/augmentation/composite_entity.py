import copy
import json
import random
import logging

from utils.logging_utils import add_color_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)


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


class CompositeEntityGenerator:
    """Add "của quân", "số 3",..."""

    def __init__(self):
        self.entity_numbers = list(range(1, 100))
        self.avail_owner = [
            "quân", "nam", "my", "trung", "hùng", "dũng", "lan", "thuỷ",
            "nguyên", "lâm", "trang", "linh", "sơn", "mạnh", "quyền", "quyết",
            "vượng", "ông bà", "bố mẹ", "long", "chiến", "tú"
        ]
        self.avail_side = [
            "phải", "trái", "trên", "dưới", "trong", "ngoài", "cạnh", "kia", "này",
            "tả", "hữu", "đông", "tây", "nam", "bắc", "tây nam", "đông nam", "tây bắc", "đông bắc",
            "cạnh", "mạn phải", "mạn trái", "mạn trong", "mạn ngoài"
        ]

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
            for idx, item in enumerate(data):
                if idx < running_idx:
                    continue
                try:
                    tagged_sequence = list(zip(item["tokens"], item["labels"]))
                    self.tagged_sequence = tagged_sequence
                    augmentations = self.diversify()
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

    def entity_diversify(self, entity):
        variants = []
        if entity["entity_name"] == "device":
            variants.extend(self.add_owner(entity))
            variants.extend(self.add_quantity(entity))
        elif entity["entity_name"] == "location":
            variants.extend(self.add_owner(entity))
        elif entity["entity_name"] == "scene":
            variants.extend(self.scene_diversify(entity))
        return variants

    def add_owner(self, entity):
        assert entity["entity_name"] in {"device", "location"}

    def add_quantity(self, entity):
        assert entity["entity_name"] == "device"

    def scene_diversify(self, entity):
        assert entity["entity_name"] == "scene"

def main():
    augmenter = CompositeEntityGenerator()
    data = load_data()
    augmenter.augment(data)


if __name__ == "__main__":
    main()
