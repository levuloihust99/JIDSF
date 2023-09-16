import csv
import copy
import argparse
import random
import numpy as np

from tqdm import tqdm
from typing import List, Any, Text


def load_data(input_path):
    data = []
    with open(input_path, "r") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append({
                "tokens": eval(row["tokens"]),
                "labels": eval(row["labels"]),
                "segmented_tokens": eval(row["segmented_tokens"]),
                "segmented_labels": eval(row["segmented_labels"])
            })
    return data


def deflatten_index(factor: int, size: int, idx: int):
    n = idx
    digits = []
    while n != 0:
        digits.append(n % size)
        n = n // size
    if len(digits) < factor:
        digits += [0] * (factor - len(digits))
    return digits[::-1]


def sample_for_factor(factor: int, size: int):
    sample_size = size ** factor
    flat_indices = random.sample(range(sample_size), size)
    indices = [deflatten_index(factor, size, idx) for idx in flat_indices]
    return indices


def make_data_sample(data_items: List[Any]):
    out_item = {}
    for item in data_items:
        for k, v in item.items():
            if k not in out_item:
                out_item[k] = copy.deepcopy(v)
            else:
                out_item[k].extend(copy.deepcopy(v))
    out_item.update(metadata={"factor": len(data_items)})
    return out_item


def do_augment(augment_factor, data):
    data_len = len(data)
    augment_data = []
    augment_data.extend([{**item, "metadata": {"factor": 1}} for item in data])
    for factor in augment_factor:
        indices = sample_for_factor(factor, data_len)
        data_for_factor = []
        for idxs in tqdm(indices, desc="Factor #{}".format(factor)):
            data_for_factor.append(make_data_sample([data[idx] for idx in idxs]))
        augment_data.extend(data_for_factor)
    random.shuffle(augment_data)
    return augment_data


def save_data(data, output_path):
    with open(output_path, "w") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=["tokens", "labels", "segmented_tokens", "segmented_labels", "metadata"])
        csvwriter.writeheader()
        for item in tqdm(data, desc="Item written"):
            csvwriter.writerow(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--input_path", default="data/hackathon-slu-2023/ner/split/train_data.csv")
    parser.add_argument("--output_path", default="data/hackathon-slu-2023/ner/split/train_data_augmented.csv")
    parser.add_argument("--augment_factor", default=[2, 3, 4, 5], type=eval)
    args = parser.parse_args()

    random.seed(args.seed)
    data = load_data(args.input_path)
    augmented_data = do_augment(args.augment_factor, data)
    save_data(augmented_data, args.output_path)


if __name__ == "__main__":
    main()
