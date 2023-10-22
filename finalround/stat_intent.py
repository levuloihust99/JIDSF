import json
from collections import defaultdict

DATA_PATH = "final/data/train_final_20230919.jsonl"

def load_data(data_path):
    data = []
    with open(data_path) as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    return data


def main():
    data = load_data(DATA_PATH)
    intent_tracker = defaultdict(list)

    for item in data:
        intent_tracker[item["intent"]].append(item)

    all_intents = list(intent_tracker.keys())
    longest_intent_in_chars = 0
    for intent in all_intents:
        if longest_intent_in_chars < len(intent):
            longest_intent_in_chars = len(intent)
    
    for intent in intent_tracker:
        print("{{:{}s}}: {{}}".format(longest_intent_in_chars).format(intent, len(intent_tracker[intent])))


if __name__ == "__main__":
    main()
