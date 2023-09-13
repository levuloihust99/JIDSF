import json
import copy
import argparse
from plugins.data_manipulation import WordSegmenter
from utils.helpers import recursive_apply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", default="data/20230913/asr_output_base.json")
    parser.add_argument("--output_path", "-o", default="data/20230913/asr_output_base_segmented.jsonl")
    args = parser.parse_args()
    segmenter = WordSegmenter(segment_endpoint="http://localhost:8088/segment")
    data = []
    with open(args.input_path, "r") as reader:
        data = json.load(reader)
    cloned_data = copy.deepcopy(data)
    recursive_apply(cloned_data, fn=segmenter.segment)
    segmented_data = []
    for item, segmented_item in zip(data, cloned_data):
        segmented_data.append({**item, "norm_segmented": segmented_item["norm"]})
    with open(args.output_path, "w") as writer:
        for item in segmented_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()