import re
import json
import argparse

from tqdm import tqdm
from typing import List, Tuple, Text
from transformers.models.bert.tokenization_bert import BasicTokenizer

basic_tokenizer = BasicTokenizer(
    do_lower_case=True,
    strip_accents=False,
    do_split_on_punc=True
)


def separate_punctuation(words: List[Tuple[Text, Text]]):
    output_words = []
    for word, entity in words:
        _words = basic_tokenizer.tokenize(word)
        if entity.startswith("B-"):
            output_words.extend(
                [(w, ("B-" if i == 0 else "I-") + entity[2:]) for i, w in enumerate(_words)
                 for w in _words]
            )
        else:
            output_words.extend([(w, entity) for w in _words])
    return output_words


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", required=True,
                        help="Path to the original data of Hackathon SLU 2023")
    parser.add_argument("--output_path", "-o", required=True)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    pattern = r"\[(.*?)\]"
    annotated_data = []
    for item in tqdm(data):
        annotation = item["sentence_annotation"].lower()
        matches = re.finditer(pattern, annotation)
        words = []
        idx = 0
        for m in matches:
            start_idx = m.start()
            end_idx = m.end()
            words.extend([(word, "O") for word in annotation[idx : start_idx].split()])
            entity_name, entity_value = [v.strip() for v in m.group(1).split(":")]
            entity_words = [
                (word, ("B-" if i == 0 else "I-") + entity_name)
                for i, word in enumerate(entity_value.split())
            ]
            words.extend(entity_words)
            idx = end_idx
        if end_idx < len(annotation):
            words.extend([(word, "O") for word in annotation[end_idx:].split()])
        annotated_data.append(separate_punctuation(words))
    
    with open(args.output_path, "w") as writer:
        for item in tqdm(annotated_data, desc="Write"):
            tokens, labels = zip(*item)
            writer.write(json.dumps({"tokens": tokens, "labels": labels}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()