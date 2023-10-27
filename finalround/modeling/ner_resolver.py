from .ner_cont import BertNERCont, RobertaNERCont

from transformers import AutoModel


def resolve_ner_cont_model_class(model_type):
    if model_type == "bert":
        return BertNERCont
    if model_type == "roberta":
        return RobertaNERCont
    if model_type == "auto":
        return AutoModel


def resolve_ner_cont_model(model_type, model_path, **kwargs):
    model_class = resolve_ner_cont_model_class(model_type)
    return model_class.from_pretrained(model_path, **kwargs)
