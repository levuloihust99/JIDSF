import os
import json
import sanic
import asyncio
import logging
import argparse
import concurrent.futures

from typing import Text
from datetime import datetime
from sanic import Sanic
from sanic_cors import CORS

from transformers import (
    BertTokenizer,
    PhobertTokenizer
)

from utils.data_utils import WordSegmenter
from api.ner.processor import NERProcessor
from utils.training_utils import load_model
from utils.logging_utils import add_color_formatter
from model.modeling import (
    BertPosTagger,
    PhoBertPosTagger
)

app = Sanic(__name__)
CORS(app)
app.config.RESPONSE_TIMEOUT = 300

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


@app.on_request
def before_request_func(request):
    request.ctx.start_time = datetime.now()


@app.on_response
def after_response_func(request, response):
    logger.info("Total processing time: {}".format(datetime.now() - request.ctx.start_time))


@app.post("/ner")
async def extract_entities(request):
    data = request.json
    text = data["text"]
    entities = request.app.ctx.ner_processor.extract(text)
    return sanic.json({"entities": entities})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--model_type", default="phobert")
    parser.add_argument("--model_path", default="vinai/phobert-base")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--segment", type=eval, default=False)
    parser.add_argument("--segment_endpoint", default="http://localhost:8088/segment")
    args = parser.parse_args()

    tokenizer = load_tokenizer(
        tokenizer_type=args.model_type,
        tokenizer_path=args.model_path
    )
    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path
    )
    with open(os.path.join(args.model_path, "label_mappings.json"), "r") as reader:
        label_mappings = json.load(reader)

    if args.segment:
        word_segmenter = WordSegmenter(args.segment_endpoint)

    ner_processor = NERProcessor(
        ner_tokenizer=tokenizer,
        ner_model=model,
        ner_label_mappings=label_mappings,
        segmenter=word_segmenter,
        args=args
    )

    app.ctx.ner_processor = ner_processor
    app.run(host="0.0.0.0", port=args.port, single_process=True)


if __name__ == "__main__":
    main()
