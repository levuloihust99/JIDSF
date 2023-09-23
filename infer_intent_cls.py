import os
import json
import torch
import logging
import argparse
from torch import nn

from utils.data_utils import WordSegmenter
from utils.logging_utils import add_color_formatter

from transformers import (
    BertTokenizer, PhobertTokenizer,
    RobertaForSequenceClassification,
    BertForSequenceClassification
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="bert")
    parser.add_argument("--model_path", default="checkpoints/IC/checkpoint-BertForSequenceClassification-5e-05-0.995729")
    parser.add_argument("--input_path", default="results/asr_output_norm.json")
    parser.add_argument("--segment", type=eval, default=False)
    parser.add_argument("--segment_endpoint", default="http://localhost:8088/segment")
    parser.add_argument("--output_path", default="results/intent_classification.jsonl")
    parser.add_argument("--data_format", default="json", choices=["json", "jsonlines"])
    args = parser.parse_args()

    if args.model_type == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = PhobertTokenizer.from_pretrained(args.model_path)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    if args.segment is True:
        segmenter = WordSegmenter(args.segment_endpoint)
    else:
        segmenter = None
    
    model.eval()

    if args.data_format == "jsonlines":
        data = []
        with open(args.input_path) as reader:
            for line in reader:
                data.append(json.loads(line.strip()))
    else:
        with open(args.input_path) as reader:
            data = json.load(reader)

    with open(os.path.join(args.model_path, "label_mappings.json"), "r") as reader:
        tag2int = json.load(reader)
    int2tag = {v: k for k, v in tag2int.items()}

    out_data = []
    with torch.no_grad():
        for idx, item in enumerate(data):
            text = item["norm"]
            if segmenter:
                text = segmenter.segment(text)
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
            logger.info("Done #{}".format(idx))

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
