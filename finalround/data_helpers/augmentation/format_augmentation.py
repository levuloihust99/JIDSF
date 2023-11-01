import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_file", default="data/train_final_20230919.jsonl")
    parser.add_argument("--input_path", "-i", default="data/ner/entity_composite/composite_entity_filtered.jsonl")
    parser.add_argument("--output_path", "-o", default="data/ner/entity_composite/composite_entity_formatted.jsonl")
    args = parser.parse_args()

    data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    mapping_data = []
    with open(args.mapping_file, "r") as reader:
        for line in reader:
            mapping_data.append(json.loads(line.strip()))
    mapping = {item["file"]: item for item in mapping_data}

    out_data = []
    for item in data:
        tagged_sequence = item["tagged_sequence"]
        tokens, labels = list(zip(*tagged_sequence))
        file = item["file"]
        out_data.append({
            "file": file,
            "tokens": tokens,
            "labels": labels,
            "text": " ".join(tokens),
            "intent": mapping[file]["intent"]
        })

    with open(args.output_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
