import re
import json
import requests

import unidecode
import copy
import jsonlines

from typing import Text, List


def taggedseq2doccano(sentences):
    error_count = 0
    dataset = []
    for i, sentence in enumerate(sentences):
        words, labels = sentence["tokens"], sentence["labels"]
        labels_p = []
        idx = 0
        while idx < len(words):
            if labels[idx].startswith('B-'):
                actual_label = labels[idx][2:]
                start_pos = idx
                for j in range(idx): start_pos += len(words[j])

                end_pos = start_pos + len(words[idx]) + 1
                j = idx + 1
                while j < len(words):
                    if labels[j] == 'I-' + actual_label:
                        end_pos += len(words[j]) + 1
                        j += 1
                    else: break
                end_pos -= 1
                labels_p.append([start_pos, end_pos, actual_label])
                idx = j
            elif labels[idx].startswith('I-'): 
                error_count += 1
                break
            else: idx += 1
        dataset.append({
            'text': " ".join(words),
            'label': labels_p
        })
    return dataset, error_count


def readJsonL(filename: str):
    """Read doccano annotated data in jsonl format
    """

    container = []
    with jsonlines.open(filename, 'r') as reader:
        for record in reader:
            if "text" not in record and "data" in record:
                record["text"] = record.pop("data")
            if "labels" not in record and "label" in record:
                record["labels"] = record.pop("label")
            container.append(record)
    return container


def doccano2tagged(filename: str, data=None):
    """Convert doccano dataset to the format for training NER
    """
    if int(bool(filename)) + int(bool(data)) != 1:
        raise Exception("Exactly `filename` or `data` must be not None.")
    if filename:
        data = readJsonL(filename)

    tagged_sequences = []
    for doc in data:
        text = doc['text']
        labels = doc['labels']
        labels.sort(key=lambda x: x[0])
        labels_iter = iter(labels)

        tagged_seq = []
        idx = 0
        while True:
            # tag the part that is front of an entity
            try:
                label = next(labels_iter)
            except StopIteration:
                break
            words = text[idx : label[0]].split()
            tagged_seq.extend([(word, 'O') for word in words])
            idx = label[0]
            
            # tag the entity
            words = text[idx : label[1]].split()
            labeled_words = [(word, "I-" + label[2]) for word in words]
            labeled_words[0] = (words[0], "B-" + label[2])
            tagged_seq.extend(labeled_words)
            idx = label[1]

        words = text[idx:].split()
        tagged_seq.extend([(word, 'O') for word in words])

        tagged_sequences.append(tagged_seq)
    
    return tagged_sequences


def segment_word(sentences, segmenter, error_tracker=None):
    """
    Segment vietnamese sentence into multi-syllable words.

    Args:
        sentences: List of tagged sentences, each sentence is a list of tagged tokens.
            E.g. [('Bệnh', 'B-ten_benh'), ('ung', 'I-ten_benh'), ('thư', 'I-ten_benh'), ('rất', 'O'), ('nguy', 'O'), ('hiểm', 'O')]
        segmenter: RDRSegmenter. Follow https://github.com/VinAIResearch/PhoBERT to see guidance for instantiating RDRSegmenter
        error_tracker: A dictionary to store index of error sentence.

    Return:
        List of tagged segmented sentences, each sentence is a list of tagged segmented tokens.
            E.g. [('Bệnh', 'B-ten_benh'), ('ung_thư', 'I-ten_benh'), ('rất', 'O'), ('nguy', 'O'), ('hiểm', 'O')]
    """
    if error_tracker is not None:
        error_tracker['idxs'] = []
    res_sentences = [] # output tagged sentences, the same format as input sentences
    error_count = 0
    
    for outer_idx, sent in enumerate(sentences): # outer_idx for debug
        error_flag = False
        tokens, labels = zip(*sent)
        tokens = list(tokens)
        labels = list(labels)
        origin_tokens = copy.deepcopy(tokens) # for debug

        origin_sentence = " ".join(tokens)
        segmented_tokens = segmenter.tokenize(origin_sentence)[0]
        segmented_tokens = [token.strip("_") for token in segmented_tokens]
        
        res_sentence = []
        marked = 0 # track the current position in `tokens` variable
        for inner_idx, segmented_token in enumerate(segmented_tokens): # inner_idx for debug
            while True: # ignore wrong empty tagged tokens. E.g. [('', 'B-PERSON')]
                # equivalent token in non-segmented sentence of `segmented_token`
                # E.g. origin sentence: Thành phố Hà Nội
                #   segmented sentence: Thành_phố Hà_Nội
                # --> segmented_token = 'Thành_phố' and aligned_token = 'Thành'
                aligned_token = tokens[marked]
                if aligned_token:
                    break
                marked += 1

            # process one-to-multi case, i.e. one origin token becomes multiple segmented tokens
            #    E.g. origin sentence: "(Đợi hoài)"
            #    non-segmented tokens: ["(Đợi", "hoài)"]
            #        segmented tokens: ["(", "Đợi", "hoài_)"]
            #           aligned token: "(Đợi"
            #         segmented token: "("
            if len(segmented_token) < len(aligned_token): # OK
                res_sentence.append((segmented_token, labels[marked]))
                frontword = aligned_token[:len(segmented_token)]
                backword = aligned_token[len(segmented_token):]
                del tokens[marked]
                labels.insert(marked, labels[marked])
                tokens.insert(marked, backword)
                tokens.insert(marked, frontword)
                marked += 1
                continue

            # process one-to-one case
            if len(segmented_token) == len(aligned_token) and \
                unidecode.unidecode(segmented_token) == unidecode.unidecode(aligned_token): # OK
                res_sentence.append((segmented_token, labels[marked]))
                marked += 1
                continue
            
            # multi-to-one case, i.e. several origin tokens are combined into one segmented token
            atomic_tokens = segmented_token.split("_") # E.g. segmented_token = 'Lê_Văn_Lương' --> atomic_tokens = ['Lê', 'Văn', 'Lương']
            token_labels = []
            L = len(atomic_tokens)
            for i in range(L):
                while True: # ignore wrong empty tagged tokens. E.g. [('', 'B-PERSON')]
                    aligned_token = tokens[marked]
                    if aligned_token:
                        break
                    marked += 1
                if atomic_tokens[i] == aligned_token: # 1. the atomic token equals the aligned token
                    token_labels.append(labels[marked])
                # 2. the same as (1) but for processing accents-shift case: 'hòa' vs 'hoà'
                elif unidecode.unidecode(aligned_token) == unidecode.unidecode(atomic_tokens[i]):
                    token_labels.append(labels[marked])
                # 3. E.g. origin sentence: "Ngày hôm nay, tính đến 6h00, Hà Nội ghi nhận 15 ca mắc mới"
                #    non-segmented tokens: ["Ngày", "hôm", "nay,",...]
                #        segmented tokens: ["Ngày", "hôm_nay", ",", ...]
                #           atomic_tokens: ["hôm", "nay"]
                #           aligned_token: "nay,"
                elif aligned_token.startswith(atomic_tokens[i]) or \
                    unidecode.unidecode(aligned_token).startswith(unidecode.unidecode(atomic_tokens[i])): # process accents-shift case
                    # split the aligned token, i.e. "(Đợi" --> ["(", "Đợi"]
                    frontword = aligned_token[:len(atomic_tokens[i])]
                    backword = aligned_token[len(atomic_tokens[i]):]
                    # change the origin tokens, i.e. alternate the aligned token by the splitted tokens
                    del tokens[marked]
                    tokens.insert(marked, backword)
                    tokens.insert(marked, frontword)
                    # change origin labels
                    lb = labels[marked]
                    labels.insert(marked + 1, lb)
                    token_labels.append(lb)
                else:
                    error_count += 1
                    print("Encounter {} errors".format(error_count))
                    error_flag = True
                    if error_tracker is not None:
                        error_tracker['idxs'].append(outer_idx)
                    break
                marked += 1

            if error_flag:
                break

            # 1. wrong case: one segmented token containing atomic tokens with more than 2 labels
            # E.g. segmented token: "Nguyễn_Văn_Nam"
            # tagged atomic tokens: [("Nguyễn", "B-PERSON"), ("Văn", "I-PERSON"), ("Nam", "O")]
            # Keep the token labels untouched, using origin tokens instead of segmented token   
            if len(set(token_labels)) > 2:
                res_sentence.extend(list(zip(
                    atomic_tokens,
                    token_labels
                )
            ))
            else:
                begin_tag = token_labels[0]
                remain_tags = set(token_labels[1:])
                remain_tag = next(iter(remain_tags), None)
                if remain_tag is None:
                    res_sentence.append((segmented_token, begin_tag))
                elif begin_tag[2:] == remain_tag[2:]: # success case: E.g. Hà_Nội --> [("Hà", "B-LOC"), ("Nội", "I-LOC")]
                    res_sentence.append((segmented_token, begin_tag))
                else: # wrong case: E.g. Hà_Nội --> [("Hà", "B-LOC"), ("Nội", "I-PERSON")]
                    res_sentence.extend(list(zip(
                        atomic_tokens,
                        token_labels
                    )))

        if error_flag:
            continue
        res_sentences.append(res_sentence)

    return res_sentences


class WordSegmenter:
    def __init__(self, segment_endpoint):
        self.segment_endpoint = segment_endpoint

    def tokenize(self, text: Text) -> List[Text]:
        return [self.segment(text).split()]

    def segment(self, text: Text) -> Text:
        headers = {"Content-Type": "application"}
        response = requests.post(self.segment_endpoint, headers=headers, data=json.dumps({"sentence": text}))
        return response.json()["sentence"]


def recursive_apply(data, fn, ignore_keys=None):
    """Hàm áp dụng hồi quy function {fn} vào data

    Args:
        data (Dict/List): _description_
        fn (function): _description_
        ignore_keys (_type_, optional): Key của Dict không áp dụng function. Defaults to None.

    Returns:
        data: _description_
    """
    stack = [(None, -1, data)]  # parent, idx, child: parent[idx] = child
    while stack:
        parent_node, index, node = stack.pop()
        if isinstance(node, list):
            stack.extend(list(zip([node] * len(node), range(len(node)), node)))
        elif isinstance(node, dict):
            stack.extend(
                list(zip([node] * len(node), node.keys(), node.values())))
        elif isinstance(node, str):
            if node and (ignore_keys is None or index not in ignore_keys):
                parent_node[index] = fn(node)
            else:
                parent_node[index] = node
        else:
            continue
    return data