import argparse


def create_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--hparams", default="{}")
    return parser


def add_intent_params(parser: argparse.ArgumentParser):
    parser.add_argument("--intent_tokenizer_type")
    parser.add_argument("--intent_tokenizer_path")
    parser.add_argument("--intent_model_type")
    parser.add_argument("--intent_model_path")

def add_ner_params(parser: argparse.ArgumentParser):
    parser.add_argument("--ner_tokenizer_type")
    parser.add_argument("--ner_tokenizer_path")
    parser.add_argument("--ner_model_type")
    parser.add_argument("--ner_model_path")
    return parser
