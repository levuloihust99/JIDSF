import os
import json
import torch
import argparse
from torch import nn

from plugins.data_manipulation import WordSegmenter

from transformers import (
    BertTokenizer, PhobertTokenizer,
    RobertaForSequenceClassification,
    BertForSequenceClassification
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="roberta")
    parser.add_argument("--model_path", default="checkpoints/intent_cls/checkpoint-RobertaForSequenceClassification-5e-05-0.996")
    parser.add_argument("--input_path", default="data/20230913/asr_output_base_segmented.jsonl")
    parser.add_argument("--segment", type=eval, default=False)
    parser.add_argument("--segment_endpoint", default="http://localhost:8088/segment")
    parser.add_argument("--output_path", default="data/20230913/public_intent_classification.jsonl")
    args = parser.parse_args()

    if args.model_type == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = PhobertTokenizer.from_pretrained(args.model_path)
    else:
        return
    
    if args.segment is True:
        segmenter = WordSegmenter(args.segment_endpoint)
    else:
        segmenter = None
    
    model.eval()

    data = []
    with open(args.input_path) as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    with open(os.path.join(args.model_path, "label_mappings.json"), "r") as reader:
        tag2int = json.load(reader)
    int2tag = {v: k for k, v in tag2int.items()}

    out_data = []
    with torch.no_grad():
        for idx, item in enumerate(data):
            text = item["norm_segmented"]
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(input_ids=inputs.input_ids, return_dict=True)
            logits = outputs.logits
            labels = torch.argmax(logits, dim=-1)
            label = labels[0].item()
            intent = int2tag[label]
            out_data.append({
                "intent": intent,
                "file": item["file_name"]
            })
            print("Done #{}".format(idx))
    
    with open(args.output_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
