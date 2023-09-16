import json
import requests
import argparse


def load_data(data_path):
    with open(data_path) as reader:
        data = json.load(reader)
    return data


def get_entities(ner_endpoint, text):
    headers = {"Content-Type": "application/json"}
    response = requests.post(ner_endpoint, data=json.dumps({"text": text}), headers=headers)
    entities = response.json()["entities"]
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner_endpoint", "-e", default="http://localhost:5577/ner")
    parser.add_argument("--data_path", "-i", default="data/20230913/asr_output_base.json")
    parser.add_argument("--output_path", "-o", default="data/20230916/public_submission_NER_debug.jsonl")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    inp_data = load_data(args.data_path)
    out_data = []
    for idx, item in enumerate(inp_data):
        out_item = {}
        out_item["file"] = item["file_name"]
        entities = get_entities(args.ner_endpoint, item["norm"].strip("."))
        out_entities = []
        for entity in entities:
            if not args.debug:
                out_entities.append({"type": entity["entity"], "filler": entity["value"]})
            else:
                out_entities.append(entity)
        out_item["entities"] = out_entities
        out_data.append(out_item)
        print("Done #{}".format(idx))
    with open(args.output_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
