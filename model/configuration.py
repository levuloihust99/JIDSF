import json


DATA_CONFIG_FILE = "configs/data_configs.json"


class NERConfig(object):
    def __init__(self, **hparams):
        # Model and data settings
        self.model_path = 'pretrained/NlpHUST/vibert4news-base-cased'
        self.tokenizer_path = 'pretrained/NlpHUST/vibert4news-base-cased'
        self.model_save = 'checkpoints' # 
        self.data = 'ner-multisources' # one of 'ner-multisources', 'ner-multisources-reduce', ner-vlsp-2018, 
                                       # 'ner-covid19-vinai', 'ner-cistailab-2021', 'ner-cistailab-2022' 

        # default training settings
        self.custom_train = False
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.max_grad_norm = 1.0
        self.dropout_prob = 0.1
        self.use_crf = False
        self.weight_decay = 0.1
        self.adam_epsilon = 1e-8
        self.learning_rate = 5e-5
        self.num_train_epochs = 10
        self.gradient_accumulation_steps = 1 # Number of updates steps to accumulate
                                             # before performing a backward/update pass.
        self.warmup_proportion = 0.1 # Proportion of training to perform 
                                     # linear learning rate warmup for.
        self.max_steps = -1 # If > 0: set total number of training steps 
                            # to perform. Override num_train_epochs.
        self.warmup_steps = 0 # Linear warmup over warmup_steps.
        self.save_checkpoints = False # Whether to save checkpoints during training.

        # training settings to be varied in experiments.
        self.max_seq_length = 256
        self.pool_type = 'concat' # one of 'concat', 'average'
        self.ignore_index = 0 # If = 0: ignore special tokens ([CLS], [SEP]) 
                              # when calculating loss. Set < 0 to keep all tokens.
        self.add_special_tokens = True # Whether to add [CLS] and [SEP]. If False,
                                       # `ignore_index` will have no effects.
        self.use_dice_loss = False # Whether to use dice loss for class imbalance.
        self.num_hidden_layer = 1 # Number of hidden layers to be aggregated in the
                                  # final embeddings.
        self.use_word_segmenter = True # Whether to use RDRSegmenter before
                                        # tokenization. Should be set to True
                                        # when using PhoBERT
        self.seed = None
        self.gpu_id = 0

        self._override_defaults(**hparams)
        self._get_data_paths()
    
    def _override_defaults(self, **hparams):
        for k, v in hparams.items():
            if k not in self.__dict__:
                raise Exception("Parameter `{}` is not defined for this model.".format(k))
            self.__dict__[k] = v

    def _get_data_paths(self):
        with open(DATA_CONFIG_FILE, 'r') as reader:
            data_configs = json.load(reader)
        self.path_to_data = data_configs[self.data]

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as reader:
            config = json.load(reader)
        
        return cls.from_dict(dictionary=config)
