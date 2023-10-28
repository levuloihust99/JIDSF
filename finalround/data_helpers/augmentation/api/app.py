import os
import json
import copy
import sanic
import torch
import random
import asyncio
import logging
import argparse
import unicodedata
import concurrent.futures
import torch.nn.functional as F

from typing import Text
from datetime import datetime
from sanic import Sanic
from sanic_cors import CORS

from transformers import (
    BertTokenizer,
    PhobertTokenizer,
    BertForMaskedLM,
    RobertaForMaskedLM
)

from utils.logging_utils import add_color_formatter
from finalround.tokenization.resolver import resolve_tokenizer
from finalround.data_helpers.augmentation.lm_diversify import Diversifier
from finalround.utils.vietnamese_words.trie import Trie

VIETNAMESE_WORDS_PATH = "finalround/utils/vietnamese_words/assets/words.txt"
vietnamese_words = []
with open(VIETNAMESE_WORDS_PATH, "r") as reader:
    for line in reader:
        line = line.strip()
        if line:
            vietnamese_words.append(line)

trie = Trie()
for word in vietnamese_words:
    trie.add(word)


app = Sanic(__name__)
CORS(app)
app.config.RESPONSE_TIMEOUT = 300

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formatter(logging.root)


def resolve_model(model_type, model_path):
    if model_type == "bert":
        model_class = BertForMaskedLM
    elif model_type == "roberta":
        model_class = RobertaForMaskedLM
    else:
        raise Exception("ERROR")
    model = model_class.from_pretrained(model_path)
    model.eval()
    return model


def get_all_idxs_sequence(*dims):
    if not dims:
        return [tuple()]
    follow_idxs_sequence = get_all_idxs_sequence(*dims[1:])
    all_idxs_sequence = []
    for i in range(dims[0]):
        for idxs in follow_idxs_sequence:
            all_idxs_sequence.append((i,) + idxs)
    return all_idxs_sequence


@app.post("/fill_mask")
def fill_mask_token(request):
    data = request.json
    topk = data.get("topk", 3)
    text = data.get("text", None)
    if text is None:
        return sanic.json({"error": "Missing 'text'"}, status=400)
    exclude_words = data.get("exclude_words", [])

    tokens = tokenizer.tokenize(text)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
    active_mask = input_ids.squeeze().eq(tokenizer.mask_token_id) # [seq_len]
    logits = outputs.logits.squeeze(dim=0)[active_mask] # [num_mask, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1) # [num_mask, vocab_size]

    exclude_mask = None
    if exclude_words:
        exclude_mask = []
        for word in exclude_words:
            if word in tokenizer.vocab:
                exclude_mask.append(tokenizer.vocab[word])
        exclude_mask = torch.tensor(exclude_mask)
    
    this_vocab_mask = vocab_mask.clone()
    if exclude_mask is not None:
        this_vocab_mask[exclude_mask] = True
    log_probs[:, this_vocab_mask] = -1e20

    all_mask_scores, all_mask_pred_cands = torch.topk(log_probs, topk, dim=-1) # [num_mask, topk]
    idxs_sequence = get_all_idxs_sequence(*([topk] * all_mask_scores.size(0)))
    cand_sequence = []
    for seq in idxs_sequence:
        seq_score = []
        seq_token_ids = []
        reduced_seq_score = 0.0
        for i, j in enumerate(seq):
            token_score = all_mask_scores[i][j].item()
            reduced_seq_score += token_score
            seq_score.append(token_score)
            seq_token_ids.append(all_mask_pred_cands[i][j].item())
        cand_sequence.append({
            "seq_score": seq_score,
            "reduced_seq_score": reduced_seq_score,
            "seq_token_ids": seq_token_ids,
        })

    cand_sequence = sorted(cand_sequence, key=lambda x: x["reduced_seq_score"], reverse=True)
    topk_cand_sequence = cand_sequence[:topk]

    all_output = []
    for seq in topk_cand_sequence:
        mask_idx = 0
        ignored_idxs = []
        out_tokens = copy.deepcopy(tokens)
        selected_words = tokenizer.convert_ids_to_tokens(seq["seq_token_ids"])
        for idx, word in enumerate(out_tokens):
            if word == tokenizer.mask_token:
                if selected_words[mask_idx] is not None:
                    out_tokens[idx] = selected_words[mask_idx]
                    mask_idx += 1
                    if mask_idx == len(selected_words):
                        break
                else:
                    ignored_idxs.append(idx)
                    out_tokens[idx] = tokenizer.unk_token
        all_output.append({
            "output": tokenizer.decode(
                tokenizer.convert_tokens_to_ids(out_tokens),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ),
            "scores": seq["seq_score"],
            "reduced_score": seq["reduced_seq_score"]
        })
    
    return sanic.json(all_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--model_type", default="bert")
    parser.add_argument("--model_path", default="NlpHUST/vibert4news-base-cased")
    parser.add_argument("--tokenizer_type", default="bert")
    parser.add_argument("--tokenizer_path", default="NlpHUST/vibert4news-base-cased")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--segment", type=eval, default=False)
    parser.add_argument("--lower", default=False, action="store_true")
    parser.add_argument("--segment_endpoint", default="http://localhost:8088/segment")
    args = parser.parse_args()

    global tokenizer, model, vocab_mask
    tokenizer = resolve_tokenizer(
        tokenizer_type=args.tokenizer_type,
        tokenizer_path=args.tokenizer_path
    )

    vocab_mask = [False] * tokenizer.vocab_size
    for word in vietnamese_words:
        if word in tokenizer.vocab:
            vocab_mask[tokenizer.vocab[word]] = True
    vocab_mask = (1 - torch.tensor(vocab_mask).to(torch.long)).to(torch.bool)

    model = resolve_model(
        model_type=args.model_type,
        model_path=args.model_path
    )

    return args


if __name__ == "__main__":
    args = main()
    app.run(host="0.0.0.0", port=args.port, single_process=True)
