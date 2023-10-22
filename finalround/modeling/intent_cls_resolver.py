from .intent_cls import BertIntentClassifier, RobertaIntentClassifier

from transformers import AutoModel


def resolve_intent_cls_model_class(model_type):
    if model_type == "bert":
        return BertIntentClassifier
    if model_type == "roberta":
        return RobertaIntentClassifier
    if model_type == "auto":
        return AutoModel


def resolve_intent_cls_model(model_type, model_path, **kwargs):
    model_class = resolve_intent_cls_model_class(model_type)
    return model_class.from_pretrained(model_path, **kwargs)
