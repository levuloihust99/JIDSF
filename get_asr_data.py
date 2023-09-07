import os
import json
import requests

from urllib3.filepost import encode_multipart_formdata
from urllib3.fields import RequestField, guess_content_type

ASR_ENDPOINT = "http://localhost:8907/speech_to_text"
# PUBLIC_TEST_DIR = "data/public_test"
TRAIN_AUDIO_DIR = "data/public_test_16k_1ac_vad"

def make_request(file_path):
    with open(file_path, "rb") as reader:
        content = reader.read()

    filename = os.path.basename(file_path)
    rf = RequestField(name="content", data=content, filename=filename)
    rf.make_multipart(content_type=guess_content_type(filename))

    payload = [
        rf,
        ("is_normalize", "1")
    ]
    body, content_type = encode_multipart_formdata(payload)
    response = requests.post(ASR_ENDPOINT, headers={
        'Content-Type': content_type,
        "api_key": os.environ["API_KEY"]
    }, data=body)

    data = response.json()
    return {
        "raw": data["raw_output"],
        "norm": data["response"]
    }


def main():
    filenames = []
    with open("data/public_test_files.txt", "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                filenames.append(line)

    # train_data = []
    # with open("data/train_verified.jsonl") as reader:
    #     for line in reader:
    #         train_data.append(json.loads(line.strip()))
    # filenames = [item["file"] for item in train_data]

    output_path = "data/asr_public_test_20230907.json"

    if os.path.exists(output_path):
        with open(output_path) as reader:
            outputs = json.load(reader)
    else:
        outputs = [None] * len(filenames)
    for idx, filename in enumerate(filenames):
        if outputs[idx]:
            print("Skip #{}".format(idx))
            continue
        file_path = os.path.join(TRAIN_AUDIO_DIR, filename)
        try:
            asr_output = make_request(file_path)
            outputs[idx] = {"file_name": filename, **asr_output}
            print("Successful #{}".format(idx))
        except Exception as e:
            print("Exception: #{}, file name: {}".format(idx, filename))
            with open(output_path, "w") as writer:
                json.dump(outputs, writer, ensure_ascii=False, indent=4)

    with open(output_path, "w") as writer:
        json.dump(outputs, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
