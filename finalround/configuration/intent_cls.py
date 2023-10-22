import logging

logger = logging.getLogger(__name__)


class IntentClassifierConfig:
    def __init__(self, **kwargs):
        self.data_path = "data/train_final_20230919.jsonl"
        self.train_indices_path = "data/ner/train_indices.json"
        self.dev_indices_path = "data/ner/dev_indices.json"
        self.do_eval = True
        self.tokenizer_path = "pretrained/NlpHUST/vibert4news-base-cased"
        self.tokenizer_type = "bert"
        self.model_path = "pretrained/NlpHUST/vibert4news-base-cased"
        self.model_type = "bert"
        self.add_pooling_layer = True
        self.sim_func = "cosine"
        self.scale_cosine_factor = 5.0
        self.max_seq_length = 512
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.eval_steps = 100
        self.eval_metric = "micro"
        self.save_steps = 100
        self.only_save_better = True
        self.weight_decay = 0.1
        self.learning_rate = 5e-5
        self.adam_epsilon = 1e-8
        self.total_updates = 1000
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0
        self.warmup_steps = 0
        self.seed = 12345
        self.checkpoint_dir = "checkpoints/IC"

        self.override_defaults(**kwargs)
        self.validate_config()

    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v
    
    def validate_config(self):
        assert self.tokenizer_type in ["bert", "phobert", "auto"]
        assert self.model_type in ["bert", "roberta", "auto"]
        assert self.sim_func in ["cosine", "dot_product"]
        assert self.eval_metric in ["micro", "macro", "weighted"]

    def to_json(self):
        return {
            "data_path": self.data_path,
            "train_indices_path": self.train_indices_path,
            "dev_indices_path": self.dev_indices_path,
            "do_eval": self.do_eval,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_type": self.tokenizer_type,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "add_pooling_layer": self.add_pooling_layer,
            "sim_func": self.sim_func,
            "scale_cosine_factor": self.scale_cosine_factor,
            "max_seq_length": self.max_seq_length,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "eval_steps": self.eval_steps,
            "eval_metric": self.eval_metric,
            "save_steps": self.save_steps,
            "only_save_better": self.only_save_better,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "total_updates": self.total_updates,
            "warmup_proportion": self.warmup_proportion,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "seed": self.seed,
            "checkpoint_dir": self.checkpoint_dir
        }