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

    parser.add_argument('--model-path')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--model-save')
    parser.add_argument('--data', choices=['ner-multisources', 'ner-multisources-reduce', 'ner-covid19-vinai',
                                           'ner-vlsp-2018', 'ner-cistailab-2021', 'ner-cistailab-2022'])


def add_training_params(parser):
    """Params for training settings."""

    parser.add_argument('--custom-train', type=eval)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--max-grad-norm', type=float)
    parser.add_argument('--dropout-prob', type=float)
    parser.add_argument('--use-crf', type=eval)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--adam-epsilon', type=float)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--num-train-epochs', type=int)
    parser.add_argument('--gradient-accumulation-steps', type=int)
    parser.add_argument('--warmup-proportion', type=float)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--save-checkpoints', type=eval)
    parser.add_argument('--pool-type', choices=['concat', 'average'])
    parser.add_argument('--ignore-index', type=int)
    parser.add_argument('--add-special-tokens', type=eval)
    parser.add_argument('--use-dice-loss', type=eval)
    parser.add_argument('--num-hidden-layer', type=int)
    parser.add_argument('--use-word-segmenter', type=eval)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu-id', type=int)

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
