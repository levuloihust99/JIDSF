source .venv/bin/activate; python infer_intent_cls.py \
    --model_type bert \
    --model_path checkpoints/IC/checkpoint-BertForSequenceClassification-5e-05-0.995729 \
    --input_path results/asr_output_norm.json \
    --segment False