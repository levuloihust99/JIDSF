import json
import copy
import logging

from utils.utils import setup_random
from utils.logging_utils import add_color_formatter
from finalround.arguments.intent_cls_parser import create_parser, override_defaults
from finalround.configuration.intent_cls import IntentClassifierConfig
from finalround.trainer.intent_cls import IntentClassifierTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)


def main():
    parser = create_parser()
    args = parser.parse_args()

    args_json = copy.deepcopy(args.__dict__)
    hparams = args_json.pop('hparams')
    if args.hparams.endswith('.json'):
        with open(args.hparams, "r") as f:
            hparams = json.load(f)
    else:
        hparams = json.loads(args.hparams)
    hparams = override_defaults(hparams, args_json)

    config = IntentClassifierConfig(**hparams)
    setup_random(config.seed)
    trainer = IntentClassifierTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
