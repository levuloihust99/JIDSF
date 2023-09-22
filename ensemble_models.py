import os
import json
import logging
import argparse

from typing import Text

from transformers import (
    BertTokenizer,
    PhobertTokenizer
)

from utils.data_utils import WordSegmenter
from api.ner.processor import NERProcessor
from utils.logging_utils import add_color_formatter
from model.modeling import (
    BertPosTagger,
    PhoBertPosTagger
)
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)


class ModelTypeNotSupported(Exception):
    """Raise when the model_type is unknown."""


def load_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == "phobert":
        tokenizer = PhobertTokenizer.from_pretrained(tokenizer_path)
    elif tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        raise ModelTypeNotSupported("The tokenizer type of '{}' is not supported.".format(tokenizer_type))
    return tokenizer


def load_model(model_type: Text, model_path: Text):
    with open(os.path.join(model_path, "training_config.json"), "r") as reader:
        training_config = json.load(reader)
    ner_args = argparse.Namespace(**training_config)
    if model_type == "phobert":
        model = PhoBertPosTagger.from_pretrained(model_path, ner_args)
    elif model_type == "bert":
        model = BertPosTagger.from_pretrained(model_path, ner_args)
    else:
        raise ModelTypeNotSupported("The model of type '{}' is not supported.".format(model_type))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="ensemble_config.json")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--segment", type=eval, default=False)
    parser.add_argument("--lower", default=False, action="store_true")
    parser.add_argument("--segment_endpoint", default="http://localhost:8088/segment")
    parser.add_argument("--data_path", "-i", default="final/data/asr_output/asr_output_norm_duyanh.json")
    parser.add_argument("--output_path", "-o", default="final/results/NER.jsonl")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    with open(args.config_file) as reader:
        config = json.load(reader)
    tokenizer = load_tokenizer(config["model_type"], config["models"][0])
    models = []
    for model_path in config["models"]:
        models.append(load_model(config["model_type"], model_path))

    with open(os.path.join(model_path, "label_mappings.json"), "r") as reader:
        label_mappings = json.load(reader)

    if args.segment:
        word_segmenter = WordSegmenter(args.segment_endpoint)
    else:
        word_segmenter = None

    processors = []
    logger.info("Loading models to ensemble...")
    for model in models:
        processors.append(
            NERProcessor(
                tokenizer=tokenizer,
                model=model,
                label_mappings=label_mappings,
                segmenter=word_segmenter,
                args=args
            )
        )
    
    with open(args.data_path, "r") as reader:
        ner_input_data = json.load(reader)
    out_data = []
    for idx, item in enumerate(ner_input_data):
        out_item = {}
        out_item["file"] = item["file_name"]
        ensemble_input = []
        for processor in processors:
            entities = processor.extract_raw(item["norm"])
            ensemble_input.append(entities)
        
        votes = []
        for _ in range(len(ensemble_input[0][0])):
            votes.append([])
        for tokens, labels in ensemble_input:
            for i, label in enumerate(labels):
                votes[i].append(label)
        ensemble_labels = []
        for vote in votes:
            counter = Counter(vote)
            sorted_counter = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)
            ensemble_labels.append(sorted_counter[0][0])
        ensemble_results = processors[0].extract_entities(tokens, ensemble_labels)

        out_entities = []
        for entity in ensemble_results:
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