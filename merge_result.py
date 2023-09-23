import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner_result", "-n", default="results/NER.jsonl")
    parser.add_argument("--intent_result", "-i", default="results/intent_classification.jsonl")
    parser.add_argument("--output_path", "-o", default="results/predictions.jsonl")
    args = parser.parse_args()

    ner_results = []
    with open(args.ner_result) as reader:
        for line in reader:
            ner_results.append(json.loads(line.strip()))
    ner_map = {}
    files = []
    for item in ner_results:
        files.append(item["file"])
        ner_map[item["file"]] = item

    intent_results = []
    with open(args.intent_result) as reader:
        for line in reader:
            intent_results.append(json.loads(line.strip()))
    intent_map = {}
    for item in intent_results:
        intent_map[item["file"]] = item
    
    data = []
    for f in files:
        data.append({
            "intent": intent_map[f]["intent"],
            "entities": ner_map[f]["entities"],
            "file": f
        })
    with open(args.output_path, "w") as writer:
        for item in data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
