from transformers import (
    AutoTokenizer, BertTokenizer, PhobertTokenizer
)


def resolve_tokenizer_class(tokenizer_type):
    if tokenizer_type == "auto":
        return AutoTokenizer
    if tokenizer_type == "bert":
        return BertTokenizer
    if tokenizer_type == "phobert":
        return PhobertTokenizer


def resolve_tokenizer(tokenizer_type, tokenizer_path):
    tokenizer_class = resolve_tokenizer_class(tokenizer_type)
    return tokenizer_class.from_pretrained(tokenizer_path)
