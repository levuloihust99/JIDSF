import os
import json
import random
import argparse

random.seed(12345)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="final/data/ner/all.jsonl")
    parser.add_argument("--split_ratio", default=[0.8, 0.2], type=eval)
    parser.add_argument("--output_dir", default="final/data/ner")
    args = parser.parse_args()

    data = []
    with open(args.input_file, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    num_train = int(len(all_indices) * args.split_ratio[0])
    train_indices = all_indices[:num_train]
    dev_indices = all_indices[num_train:]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "train_indices.json"), "w") as writer:
        json.dump(train_indices, writer)
    with open(os.path.join(args.output_dir, "dev_indices.json"), "w") as writer:
        json.dump(dev_indices, writer)


if __name__ == "__main__":
    main()
