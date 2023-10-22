import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--data_path")
    parser.add_argument("--train_indices_path")
    parser.add_argument("--dev_indices_path")
    parser.add_argument("--do_eval", type=eval)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--tokenizer_type", choices=["auto", "bert", "phobert"])
    parser.add_argument("--model_path")
    parser.add_argument("--model_type", choices=["bert", "roberta", "auto"])
    parser.add_argument("--add_pooling_layer", type=eval)
    parser.add_argument("--sim_func", choices=["cosine", "dot_product"])
    parser.add_argument("--scale_cosine_factor", type=float)
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--eval_metric", choices=["micro", "macro", "weighted"])
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--warmup_proportion", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--hparams", default="{}")

    return parser


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams