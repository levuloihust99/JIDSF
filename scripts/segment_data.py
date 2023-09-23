import os
import json
import argparse

from tqdm import tqdm
from utils.data_utils import WordSegmenter

SEGMENT_ENDPOINT = "http://localhost:8088/segment"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", default="data/train_final_20230919.jsonl")
    parser.add_argument("--output_file", "-o", default="data/train_final_20230919_segmented.jsonl")
    args = parser.parse_args()

    segmenter = WordSegmenter(segment_endpoint=SEGMENT_ENDPOINT)
    data = []
    with open(args.input_file, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    for item in tqdm(data, desc="Segment"):
        item["sentence"] = segmenter.segment(item["sentence"])
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output_file, "w") as writer:
        for item in tqdm(data, desc="Write"):
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    