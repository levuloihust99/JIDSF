import json
import requests

NER_ENDPOINT = "http://localhost:5577/ner"
headers = {"Content-Type": "application/json"}


def load_data():
    with open("data/asr_public_test_20230907.json") as reader:
        data = json.load(reader)
    return data


def get_entities(text):
    response = requests.post(NER_ENDPOINT, data=json.dumps({"text": text}), headers=headers)
    entities = response.json()["entities"]
    return entities


def main():
    inp_data = load_data()
    out_data = []
    for idx, item in enumerate(inp_data):
        out_item = {}
        out_item["file"] = item["file_name"]
        entities = get_entities(item["norm"].strip("."))
        out_entities = []
        for entity in entities:
            out_entities.append({"type": entity["entity"], "filler": entity["value"]})
        out_item["entities"] = out_entities
        out_data.append(out_item)
        print("Done #{}".format(idx))
    with open("public_submission_NER_20230907.jsonl", "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
