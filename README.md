# INSTALLATION GUIDE
## Requirements
- Python 3.8
### 1. Create virtual environment and install requirements
```shell
$ python -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install -U pip
(.venv)$ pip install wheel
(.venv)$ pip install -r requirements.txt
```

### 2. Install VnCoreNLP
```shell
$ mkdir -p vncorenlp/models/wordsegmenter
$ wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
$ wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
$ wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
$ mv VnCoreNLP-1.1.1.jar vncorenlp/ 
$ mv vi-vocab vncorenlp/models/wordsegmenter/
$ mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

### 3. Install Java (to use vncorenlp)
```shell
$ sudo apt install default-jre
```

### 4. Download data and pretrained
- Data is available at [data](https://drive.google.com/drive/folders/1zsq77pSPEKBQAVu4wvvBw9z5muc3WvGv?usp=sharing).
- Pre-trained language models are available at [pretrained](https://drive.google.com/drive/folders/12fku70JjD8AtZUa2sgtQGgoyaTWz0GUG?usp=sharing)
- Place folder `data` and `pretrained` under root folder of the project, i.e

        .
        ├── data/
        ├── pretrained/
        ├── data_utils.py
        ├── models.py
        ├── predict.py
        ├── README.md
        ├── requirements.txt
        ├── train.py
        ├── utils.py
        └── vncorenlp/

### 4. Run training (default parameters)
```shell
$ python train.py \
    --model-path pretrained/vinai/phobert-base \
    --learning-rate 5e-5 \
    --max-grad-norm 1.0 \
    --batch-size 16 \
    --pool-type concat \
    --ignore-index 0 \
    --add-special-tokens True \
    --use-dice-loss False \
    --num-hidden-layer 1 \
    --use-word-segmenter True \
    --save-pretrained False
```
