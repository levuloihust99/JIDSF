import logging
from transformers import  DataCollatorForTokenClassification

from model.configuration import NERConfig
from model.trainer import NERTrainer
from utils.arguments import create_parser, parse_args
from utils.training_utils import (
    load_model, load_tokenizer,
    load_data, create_dataloader,
    create_optimizer_and_scheduler
)
from utils.utils import setup_random
from utils.logging_utils import add_color_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)


def main():
    # get configuration
    parser = create_parser()
    args = parse_args(parser)
    config = NERConfig(**args.__dict__)

    # setup randomness
    setup_random(config.seed)

    # load data, tokenizer and model
    train_sentences, dev_sentences, tag2int, int2tag = load_data(config)
    logger.info("Label mappings...")
    logger.info(tag2int)
    num_classes = len([lb for lb in tag2int if lb != 'DEFAULT'])
    model = load_model(config, num_classes)
    tokenizer = load_tokenizer(config, do_lower_case=False, tokenize_chinese_characters=False)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    # create train dataloader
    train_dataloader = create_dataloader(
        config=config,
        tagged_sequences=train_sentences,
        tokenizer=tokenizer,
        tag2int=tag2int,
        training=True,
    )
    
    # create dev dataloader
    dev_dataloader = create_dataloader(
        config=config,
        tagged_sequences=dev_sentences,
        tokenizer=tokenizer,
        tag2int=tag2int,
        training=False,
    )

    # create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, total_steps=len(train_dataloader) * config.num_train_epochs
    )

    # create trainer
    trainer = NERTrainer(
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        tag2int=tag2int,
        int2tag=int2tag
    )

    # training
    trainer.train()


if __name__ == "__main__":
    main()
