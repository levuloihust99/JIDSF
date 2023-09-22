import os
import json
import random
import argparse

random.seed(12345)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="final/data/ner/all.jsonl")
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--output_dir", default="final/data/ner")
    args = parser.parse_args()

    data = []
    with open(args.input_file, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)

    L = len(all_indices)
    fold_num = [L // args.num_fold] * args.num_fold
    remain = L % args.num_fold
    for i in range(remain):
        L[i] += 1

    bound = [0] + [sum(fold_num[:i]) for i in range(1, args.num_fold + 1)]

    output_dir = os.path.join(args.output_dir, "fold_indices")
    for i in range(args.num_fold):
        fold_dir = os.path.join(output_dir, str(i))
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        dev_indices = all_indices[bound[i] : bound[i + 1]]
        train_indices = all_indices[:bound[i]] + all_indices[bound[i + 1]:]
        with open(os.path.join(fold_dir, "train_indices.json"), "w") as writer:
            json.dump(train_indices, writer)
        with open(os.path.join(fold_dir, "dev_indices.json"), "w") as writer:
            json.dump(dev_indices, writer)


if __name__ == "__main__":
    main()
