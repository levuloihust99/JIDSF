import os
import json
import torch
from torch import nn

from transformers import (
    BertTokenizer, PhobertTokenizer,
    RobertaForSequenceClassification,
    BertForSequenceClassification
)

model_type = "roberta"
model_path = "checkpoints/intent_cls/20230830/checkpoint-RobertaForSequenceClassification-5e-05-0.943"


def main():
    if model_type == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = PhobertTokenizer.from_pretrained(model_path)
    else:
        return
    
    model.eval()

    data = []
    with open("data/asr_output.json") as reader:
        data = json.load(reader)

    ner_data = []
    with open("public_submission_NER.jsonl", "r") as reader:
        for line in reader:
            ner_data.append(json.loads(line.strip()))
    
    with open(os.path.join(model_path, "label_mappings.json"), "r") as reader:
        tag2int = json.load(reader)
    int2tag = {v: k for k, v in tag2int.items()}

    out_data = []
    with torch.no_grad():
        for idx, item in enumerate(data):
            text = item["norm"]
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(input_ids=inputs.input_ids, return_dict=True)
            logits = outputs.logits
            labels = torch.argmax(logits, dim=-1)
            label = labels[0].item()
            intent = int2tag[label]
            out_data.append({
                "intent": intent,
                "entities": ner_data[idx]["entities"],
                "file": ner_data[idx]["file"],
            })
            print("Done #{}".format(idx))
    
    with open("public_submission.jsonl", "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
