import json
import argparse


def create_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_io_params(parser)
    add_training_params(parser)
    add_json_params(parser)
    return parser


def add_io_params(parser):
    """Params for data/models settings"""

    parser.add_argument('--model_path')
    parser.add_argument('--tokenizer_path')
    parser.add_argument('--model_save')
    parser.add_argument('--data', choices=['ner-multisources', 'ner-multisources-reduce', 'ner-covid19-vinai',
                                           'ner-vlsp-2018', 'ner-cistailab-2021', 'ner-cistailab-2022', 'ner-hackathon-2023'])
    parser.add_argument('--data_format', choices=["csv", "jsonlines"])


def add_training_params(parser):
    """Params for training settings."""

    def parse_save_freq(freq):
        try:
            freq = int(freq)
            return freq
        except Exception as e:
            return freq

    parser.add_argument('--do_eval', type=parse_save_freq)
    parser.add_argument('-save_freq', type=int)
    parser.add_argument('--custom_train', type=eval)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--dropout_prob', type=float)
    parser.add_argument('--use_crf', type=eval)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--adam_epsilon', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_train_epochs', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--warmup_proportion', type=float)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--save_checkpoints', type=eval)
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--pool_type', choices=['concat', 'average'])
    parser.add_argument('--ignore_index', type=int)
    parser.add_argument('--add_special_tokens', type=eval)
    parser.add_argument('--use_dice_loss', type=eval)
    parser.add_argument('--num_hidden_layer', type=int)
    parser.add_argument('--use_word_segmenter', type=eval)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu_id', type=int)

def add_json_params(parser):
    """Params in json file or dictionary object."""

    parser.add_argument('--hparams', default='{}')


def parse_args(parser):
    args = parser.parse_args()
    hparams_arg = args.__dict__.pop('hparams')

    if hparams_arg.endswith('.json'):
        with open(hparams_arg, 'r') as reader:
            hparams = json.load(reader)
    else: # hparams_arg is a dictionary string
        hparams = json.loads(hparams_arg)
    
    for k, v in hparams.items():
        if k not in args.__dict__:
            raise Exception("Parameter `{}` is not defined for this model.".format(k))
        args.__dict__[k] = v
    
    return args
