import json
import argparse
from transformers import BertTokenizer, PhobertTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_type", default="bert", choices=["bert", "phobert"])
    parser.add_argument("--input_path", "-i", default="data/ner/augmented/lm/lm_augmented.jsonl")
    parser.add_argument("--output_path", "-o", default="data/ner/augmented/lm/lm_augmented_filtered.jsonl")
    args = parser.parse_args()

    data = []
    with open(args.input_path) as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    if args.tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
    else:
        tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")

    out_data = []
    for item in data:
        if item["variant"] == "replace_entity":
            continue
        tokens, labels = zip(*item["tagged_sequence"])
        text = " ".join(tokens)
        if tokenizer.unk_token in text or "##" in text or "@@" in text:
            continue
        out_data.append(item)
    
    with open(args.output_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
