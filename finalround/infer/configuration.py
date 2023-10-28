import logging

logger = logging.getLogger(__name__)


class IntentConfig:
    def __init__(self, **kwargs):
        self.tokenizer_type = "bert"
        self.tokenizer_path = "NlpHUST/vibert4news-base-cased"
        self.model_type = "bert"
        self.model_path = "pretrained/NlpHUST/vibert4news-base-cased"

        self.override_defaults(**kwargs)
        
    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v


class NERConfig:
    def __init__(self, **kwargs):
        self.tokenizer_type = "bert"
        self.tokenizer_path = "NlpHUST/vibert4news-base-cased"
        self.model_type = "bert"
        self.model_path = "pretrained/NlpHUST/vibert4news-base-cased"

        self.override_defaults(**kwargs)
        
    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v


class InferConfig:
    def __init__(self, intent_config: IntentConfig, ner_config: NERConfig):
        self.intent_config = intent_config
        self.ner_config = ner_config

    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v
