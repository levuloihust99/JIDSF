# NER

## Step 1. Prepare data

```shell
python -m scripts.prepare_ner_data
```

After running this command, two files are generated, i.e. `data/ner/train_indices.json`, `data/ner/dev_indices.json`. We provide these two files for reproducibility.

## Step 2. Run train

```shell
bash run_train_ner.sh
```

Running this command performs NER training. On our computer (Kaggle P100), we got a checkpoint `checkpoints/ner-hackathon-2023/vibert4news-base-cased/Q0JNCHRAN7RGZBZB/checkpoint-BertPosTagger-5e-05-0.994814`. We provide this checkpoint for reproducibility.

## Step 3. Infer

```shell
bash run_infer_ner.sh
```

To successfully run this command, we need ASR output to be placed at `results/asr_output_norm.json`. After running this command, a file `results/NER.jsonl` is generated.

# Intent classification

## Step 1. Run train

```shell
bash train_IC.sh
```

Running this command performs NER training. On our computer, we got a checkpoint `checkpoints/IC/checkpoint-BertForSequenceClassification-5e-05-0.995729`. We provide this checkpoint for reproducibility.

## Step 2. Infer

```shell
bash run_infer_IC.sh
```

To successfully run this command, we need ASR output to be placed at `results/asr_output_norm.json`. After running this command, a file `results/intent_classification.jsonl` is generated.

## Step 4. Merge 

```shell
python merge_result.py
```

After running this command, a file `results/predictions.jsonl` is generated, which is ready for submission.