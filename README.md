# Setup
## Prerequisites
- Ubuntu 20.04 / MacOS Ventura 13.4.1
- Python 3.9.6

## Create virtual environment
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

# Intent Classification
## Train Intent classification with Contrastive Learning
### Split train test data
```shell
python -m scripts.split_train_test --input_file data/train_final_20230919.jsonl --split_ratio "[0.8, 0.2]" --output_dir data/split
```

### Training
```shell
python -m finalround.train_intent_cls \
    --data_path data/train_final_20230919.jsonl \
    --train_indices_path data/split/train_indices.json \
    --dev_indices_path data/split/dev_indices.json \
    --do_eval True \
    --tokenizer_path NlpHUST/vibert4news-base-cased \
    --tokenizer_type bert \
    --model_path NlpHUST/vibert4news-base-cased \
    --model_type bert \
    --add_pooling_layer True \
    --sim_func cosine \
    --scale_cosine_factor 5.0 \
    --max_seq_length 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --eval_steps 100 \
    --eval_metric micro \
    --save_steps 100 \
    --only_save_better False \
    --weight_decay 0.1 \
    --learning_rate 5e-5 \
    --adam_epsilon 1e-8 \
    --total_updates 1800 \
    --warmup_proportion 0.1 \
    --max_grad_norm 1.0 \
    --warmup_steps 0 \
    --seed 12345 \
    --checkpoint_dir checkpoints/IC
```

# Named Entity Recognition
## Create augmented data
### Step 1. Convert the original data to NER-ready format
```shell
python -m scripts.prepare_ner_data -i data/train_final_20230919.jsonl -o data/ner/all.jsonl
```
### Step 2. Create LM (language model) augmentations
```shell
python -m finalround.data_helpers.augmentation.lm_diversify --data_path data/ner/all.jsonl
```
### Step 3. Create composite entity augmentations
```shell
python -m finalround.data_helpers.augmentation.composite_entity --data_path data/ner/all.jsonl
```
### Step 4. Filter augmented data
```shell
python -m finalround.data_helpers.augmentation.filter_augmentations -i data/ner/augmented/lm/lm_augmented.jsonl -o data/ner/augmented/lm/lm_augmented_filtered.jsonl
python -m finalround.data_helpers.augmentation.filter_augmentations -i data/ner/augmented/entity_composite/entity_composite.jsonl -o data/ner/augmented/entity_composite/entity_composite_filtered.jsonl
```
### Step 5. Format augmented data
```shell
python -m finalround.data_helpers.augmentation.format_augmentation -i data/ner/augmented/lm/lm_augmented_filtered.jsonl -o data/ner/augmented/lm/lm_augmented_formatted.jsonl
python -m finalround.data_helpers.augmentation.format_augmentation -i data/ner/augmented/entity_composite/entity_composite_filtered.jsonl -o data/ner/augmented/entity_composite/entity_composite_formatted.jsonl
```
### Step 6. Combine augmented data
```shell
cat data/ner/augmented/lm/lm_augmented_formatted.jsonl data/ner/augmented/entity_composite/entity_composite_formatted.jsonl > data/ner/augmented/all_augmented.jsonl
```

## Split train test data
```shell
python -m scripts.split_train_test --input_file data/ner/augmented/all_augmented.jsonl --split_ratio "[0.8, 0.2]" --output_dir data/ner/augmented/split
```

## Train NER
```shell
python train_ner.py \
    --model_path NlpHUST/vibert4news-base-cased \
    --tokenizer_path NlpHUST/vibert4news-base-cased \
    --model_save checkpoints/ner \
    --data ner-hackathon-2023 \
    --data_format jsonlines \
    --do_eval True \
    --save_freq 1000 \
    --custom_train False \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --max_grad_norm 1.0 \
    --dropout_prob 0.1 \
    --use_crf False \
    --weight_decay 0.1 \
    --adam_epsilon 1e-8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 1 \
    --warmup_proportion 0.05 \
    --max_steps -1 \
    --warmup_steps 0 \
    --save_checkpoints True \
    --max_seq_length 512 \
    --pool_type concat \
    --ignore_index -100 \
    --add_special_tokens True \
    --use_dice_loss False \
    --num_hidden_layer 1 \
    --use_word_segmenter False \
    --seed 12345
```

# Inference
## Prerequisites
* A file `results/results_private_180000.json` is available.
* A NER checkpoint `checkpoints/ner/checkpoint-BertPosTagger-5e-05-0.998212`.
* An intent classification checkpoint `checkpoints/intent_cls/checkpoint-BertIntentClassifier-5e-05-micro_f1_0.997997`.
* A file `ensemble_config.json` with the following content
<pre>
{
    "model_type": "bert",
    "models": [
        "checkpoints/ner/checkpoint-BertPosTagger-5e-05-0.998212"
    ]
}
</pre>

## Step 1. Infer NER
```shell
python ensemble_models.py --lower --data_path results/results_private_180000.json --config_file ensemble_config.json --output_path results/NER.jsonl
```
## Step 2. Infer intent classification
```shell
python infer_intent_cls.py \
    --model_type bert \
    --model_path checkpoints/intent_cls/checkpoint-BertIntentClassifier-5e-05-micro_f1_0.997997 \
    --input_path results/results_private_180000.json \
    --output_path results/intent_classification.jsonl
    --segment False
```
## Step 3. Merge NER and intent classification result
```shell
python merge_result.py \
    --intent_result results/intent_classification.jsonl \
    --ner_result results/NER.jsonl \
    --output_path results/predictions_raw.jsonl
```
## Step 4. Post processing using rules
```shell
python -m finalround.data_helpers.augmentation.rule_fixer \
    --inp_predict_path results/predictions_raw.jsonl \
    --out_predict_path results/predictions.jsonl
```
