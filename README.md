# GUIDE
## Requirements
- Python 3.8
### 1. Create virtual environment and install requirements
```shell
$ python -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install -r requirements.txt
```

## Train NER

### <a name="data-preparation"></a> 1. Data preparation
Download data at [Hackathon-2023-SLU](https://drive.google.com/file/d/1MqG7cUIc8XMQNeTK-pzoaKqR65l2UUQF/view?usp=sharing) and extract to the root directory of this repository.

### 2. Train
<pre>
python train.py \
    --model_path vinai/phobert-base \
    --tokenizer_path vinai/phobert-base \
    --model_save checkpoints \
    --data ner-hackathon-2023 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --save_checkpoints True \
    --max_seq_length 256 \
    --pool_type concat \
    --warmup_proportion 0.05 \
    --ignore_index 0 \
    --num_hidden_layer 1 \
    --add_special_tokens True \
    --use_word_segmenter True \
    --seed 12345 \
    --gpu_id 0
</pre>

## Train Intent Classifier
### 1. Data preparation
See [previous section](#data-preparation)
### 2. Train
```shell
python train_intent_classifier.py
```

## Infer NER
### 1. Run NER API

**Prerequisites**: Java Development Kit (JDK)

#### Run segment endpoint
```shell
git clone https://github.com/levuloihust99/JavaSegmentationServer.git
cd JavaSegmentationServer
javac -d bin server/JsonServer.java
java -cp bin server.JsonServer
```
The segment endpoint runs at port 8088 by default.

#### Run NER API
```shell
python -m api.serve \
    --model_type phobert \ # if using phobert
    --model_path /path/to/the/trained/NER/model \
    --segment True \ # if using phobert
    --segment_endpoint http://localhost:8088/segment
```

#### Prepare input data for extract NER

* Call ASR api to convert audio to text

* The result should be saved in a `.json` file with the following format:
<pre>
[
    {
        "file_name": Audio file name,
        "raw": ASR result before normalization,
        "norm": ASR result after normalization
    }
]
</pre>

E.g.

<pre>
[
    {
        "file_name": "01MrFDhm4z0EUqG9IKlTGSt.wav",
        "raw": "bộ cho anh em hàn cửa số mười tới",
        "norm": "Bộ cho anh em hàn cửa số 10 tới."
    }
]
</pre>

Suppose the above data is saved in `data/asr_public_test_20230907.json`, run the following script to extract entities.

```shell
python extract_entity.py
```

NER output is saved in `public_submission_NER_20230907.jsonl`

## Infer intent classification
```shell
python infer_intent_cls.py
```

Running this command generates a `predictions.jsonl` file. Zip this file, rename to `Submission.zip` and submit.