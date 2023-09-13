import os
import json
import requests
import argparse

from urllib3.filepost import encode_multipart_formdata
from urllib3.fields import RequestField, guess_content_type


def make_request(asr_endpoint, file_path):
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
    response = requests.post(asr_endpoint, headers={
        'Content-Type': content_type,
        "api_key": os.environ["API_KEY"]
    }, data=body)

    data = response.json()
    return {
        "raw": data["raw_output"],
        "norm": data["response"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_endpoint", "-e", default="http://localhost:8908/speech_to_text")
    parser.add_argument("--audio_dir", "-i", default="data/public_test")
    parser.add_argument("--output_path", "-o", default="data/20230913/asr_output_base.json")
    args = parser.parse_args()

    filenames = os.listdir(args.audio_dir)

    if os.path.exists(args.output_path):
        with open(args.output_path) as reader:
            outputs = json.load(reader)
    else:
        outputs = [None] * len(filenames)
    for idx, filename in enumerate(filenames):
        if outputs[idx]:
            print("Skip #{}".format(idx))
            continue
        file_path = os.path.join(args.audio_dir, filename)
        try:
            asr_output = make_request(args.asr_endpoint, file_path)
            outputs[idx] = {"file_name": filename, **asr_output}
            print("Successful #{}".format(idx))
        except Exception as e:
            print(e)
            print("Exception: #{}, file name: {}".format(idx, filename))
            with open(args.output_path, "w") as writer:
                json.dump(outputs, writer, ensure_ascii=False, indent=4)

    with open(args.output_path, "w") as writer:
        json.dump(outputs, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
