import re
import copy
import json
import argparse

from collections import defaultdict


def fix_muc_do(data):
    data = copy.deepcopy(data)
    count = 0
    for item in data:
        intent = item["intent"]
        if "%" in item["text"] and "mức độ" in item["intent"]:
            intent = item["intent"].replace("mức độ", "độ sáng")
            count += 1
        item["intent"] = intent
    print("Num fix_muc_do: {}".format(count))
    return data


def fix_number(data):
    data = copy.deepcopy(data)

    def re_fix_percent_fn(match):
        return match.group("digits")[-2:] + "%"
    
    def re_rm_space_percent(match):
        return match.group("digits") + "%"

    count = 0
    for item in data:
        for idx, entity in enumerate(item["entities"]):
            entity_value = entity["filler"]
            match = re.search(r"(?P<digits>\d+)\s*%", entity_value)
            if match:
                entity_value = re.sub(r"(?P<digits>\d+)\s*%", re_rm_space_percent, entity_value)
            match = re.search(r"(?P<digits>\d{3,})\s*%", entity_value)
            if match:
                count += 1
                entity_value = re.sub(r"(?P<digits>\d{3,})\s*%", re_fix_percent_fn, entity_value)
            entity["filler"] = entity_value
    print("Num fix number: {}".format(count))
    return data


def fix_duplicated_entities(data):
    data = copy.deepcopy(data)

    count = 0
    for item in data:
        counter = defaultdict(list)
        for entity in item["entities"]:
            counter[entity["type"]].append(entity)
        has_duplicated = False
        for k, v in counter.items():
            if len(v) > 1:
                has_duplicated = True
                break
        if has_duplicated:
            count += 1
            out_entities = []
            for entity_type in counter:
                typed_entities = counter[entity_type]
                if len(typed_entities) == 1:
                    out_entities.extend(typed_entities)
                else:
                    if entity_type in {"changing value", "target number"}:
                        for e in typed_entities:
                            if "%" in e["filler"]:
                                break
                        out_entities.append(e)
                    elif entity_type == "command":
                        filtered_typed_entities = []
                        for e in typed_entities:
                            if (
                                not e["filler"].startswith("cho") and
                                not e["filler"].startswith("làm") and
                                not e["filler"].startswith("khiến") and
                                e["filler"] not in {"lên", "hoạt động", "thấp"}
                            ):
                                filtered_typed_entities.append(e)
                        typed_entities = filtered_typed_entities
                        if len(typed_entities) == 1:
                            out_entities.extend(typed_entities)
                        else:
                            if "kiểm tra" in item["intent"]:
                                for e in typed_entities:
                                    if e["filler"] in {"kiểm tra", "check", "xem"}:
                                        break
                                out_entities.append(e)
                            elif "tăng" in item["intent"]:
                                selected_entity = typed_entities[0]
                                for e in typed_entities:
                                    if e["filler"] in {"tăng", "thêm", "nâng"}:
                                        selected_entity = e
                                        break
                                out_entities.append(selected_entity)
                            elif "giảm" in item["intent"]:
                                selected_entity = typed_entities[0]
                                for e in typed_entities:
                                    if e["filler"] in {"hạ", "bớt", "giảm"}:
                                        selected_entity = e
                                        break
                                out_entities.append(selected_entity)
                            elif "bật" in item["intent"]:
                                selected_entity = typed_entities[0]
                                for e in typed_entities:
                                    if "bật" in e["filler"]:
                                        selected_entity = e
                                        break
                                out_entities.append(selected_entity)
                            elif "tắt" in item["intent"]:
                                selected_entity = typed_entities[0]
                                for e in typed_entities:
                                    if e["filler"] in {"tắt", "ngắt", "dừng", "ngưng", "ngừng"}:
                                        selected_entity = e
                                        break
                                out_entities.append(selected_entity)
                            elif "mở" in item["intent"]:
                                selected_entity = typed_entities[0]
                                for e in typed_entities:
                                    if "mở" in e["filler"]:
                                        selected_entity = e
                                        break
                                out_entities.append(selected_entity)
                            elif "đóng" in item["intent"]:
                                selected_entity = typed_entities[0]
                                for e in typed_entities:
                                    if "đóng" in e["filler"]:
                                        selected_entity = e
                                        break
                                out_entities.append(selected_entity)
                            else:
                                selected_entity = typed_entities[0]
                                out_entities.append(selected_entity)
                    elif entity_type == "device":
                        filtered_typed_entities = []
                        for e in typed_entities:
                            if e["filler"] not in {"điện", "làm việc"}:
                                filtered_typed_entities.append(e)
                            else:
                                print("Something")
                        typed_entities = filtered_typed_entities
                        out_entities.append(typed_entities[-1])
                    else:
                        out_entities.append(typed_entities[0])
            item["entities"] = out_entities
    print("Num duplicated: {}".format(count))
    return data


def fix_ra_ngoai_and_khach_toi_nha(data):
    data = copy.deepcopy(data)

    for item in data:
        text = item["text"]
        if "ra ngoài" in text:
            entities = item["entities"]
            entity_tracker = defaultdict(list)
            for entity in entities:
                entity_tracker[entity["type"]].append(entity)
            entity_tracker["scene"] = [{"type": "scene", "filler": "ra ngoài"}]
            item["entities"] = []
            for typed_entities in entity_tracker.values():
                item["entities"].extend(typed_entities)
        if "khách tới nhà" in text:
            entities = item["entities"]
            entity_tracker = defaultdict(list)
            for entity in entities:
                entity_tracker[entity["type"]].append(entity)
            entity_tracker["scene"] = [{"type": "scene", "filler": "khách tới nhà"}]
            item["entities"] = []
            for typed_entities in entity_tracker.values():
                item["entities"].extend(typed_entities)
        if "về nhà" in text:
            entities = item["entities"]
            entity_tracker = defaultdict(list)
            for entity in entities:
                entity_tracker[entity["type"]].append(entity)
            entity_tracker["scene"] = [{"type": "scene", "filler": "về nhà"}]
            item["entities"] = []
            for typed_entities in entity_tracker.values():
                item["entities"].extend(typed_entities)
    return data


def fix_intent_base_on_command(data):
    data = copy.deepcopy(data)
    for item in data:
        entities = item["entities"]
        for entity in entities:
            if entity["type"] == "command":
                if entity["filler"] in {"chạy", "phát", "khởi động"}:
                    item["intent"] = "bật thiết bị"
                elif entity["filler"] in {"ngắt", "dừng", "ngưng", "ngừng", "sập"}:
                    item["intent"] = "tắt thiết bị"
                elif entity["filler"] in {"khởi"} and "khởi động" in item["text"]:
                    item["intent"] = "bật thiết bị"
                    i_entities = item["entities"]
                    item["entities"] = []
                    for e in i_entities:
                        if e["type"] == "command":
                            e["filler"] = "khởi động"
                        item["entities"].append(e)
    return data


def fix_filter_device(data):
    data = copy.deepcopy(data)
    for item in data:
        counter = defaultdict(list)
        for entity in item["entities"]:
            counter[entity["type"]].append(entity)

        out_entities = []
        for entity_type, typed_entities in counter.items():
            if entity_type == "command":
                filtered_typed_entities = []
                for e in typed_entities:
                    if (
                        not e["filler"].startswith("cho") and
                        not e["filler"].startswith("làm") and
                        not e["filler"].startswith("khiến") and
                        e["filler"] not in {"lên", "hoạt động"}
                    ):
                        filtered_typed_entities.append(e)
                typed_entities = filtered_typed_entities
                if len(typed_entities) == 1:
                    out_entities.extend(typed_entities)
            elif entity_type == "device":
                filtered_typed_entities = []
                for e in typed_entities:
                    if e["filler"] not in {"điện", "làm việc"}:
                        filtered_typed_entities.append(e)
                typed_entities = filtered_typed_entities
                if len(typed_entities) == 1:
                    out_entities.extend(typed_entities)
            else:
                out_entities.extend(typed_entities)
        item["entities"] = out_entities
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_result_path", default="onboard/asr_result/results_private_180000.json")
    parser.add_argument("--inp_predict_path", default="onboard/submissions/submit5/predictions_raw.jsonl")
    parser.add_argument("--out_predict_path", default="onboard/submissions/submit5/predictions.jsonl")
    args = parser.parse_args()

    with open(args.asr_result_path, "r") as reader:
        asr_result = json.load(reader)
    mapping = {item["file_name"]: item for item in asr_result}

    pred_data = []
    with open(args.inp_predict_path, "r") as reader:
        for line in reader:
            pred_data.append(json.loads(line.strip()))

    added_text_data = []
    for item in pred_data:
        added_text_data.append({
            "intent": item["intent"],
            "entities": item["entities"],
            "text": mapping[item["file"]]["norm"],
            "file": item["file"]
        })

    out_data = fix_muc_do(added_text_data)
    out_data = fix_number(out_data)
    out_data = fix_duplicated_entities(out_data)
    out_data = fix_ra_ngoai_and_khach_toi_nha(out_data)
    out_data = fix_intent_base_on_command(out_data)
    out_data = fix_filter_device(out_data)
    with open(args.out_predict_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


def public_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_predict_path", default="onboard/submissions/public/submit1/new_public_predictions.jsonl")
    parser.add_argument("--out_predict_path", default="onboard/submissions/public/submit1/fixed_new_public_predictions.jsonl")
    args = parser.parse_args()

    pred_data = []
    with open(args.inp_predict_path, "r") as reader:
        for line in reader:
            pred_data.append(json.loads(line.strip()))

    out_data = fix_muc_do(pred_data)
    out_data = fix_number(out_data)
    out_data = fix_duplicated_entities(out_data)
    with open(args.out_predict_path, "w") as writer:
        for item in out_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")    


def test():
    text = "Giảm điều hoà xuống 4521 % hộ tôi"
    match = re.search(r"(?P<digits>\d{3,})\s*%", text)
    def fix_percent_fn(match):
        return match.group("digits")[-2:] + "%"
    if match:
        text = re.sub(r"(?P<digits>\d{3,})\s*%", fix_percent_fn, text)
    print("done")


def test2():
    data = []
    with open("onboard/submissions/submit1/duplicated_entities.jsonl", "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    data = fix_duplicated_entities(data)
    with open("onboard/submissions/submit1/deduplicated_entities.jsonl", "w") as writer:
        for item in data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    # test2()
    # public_main()
