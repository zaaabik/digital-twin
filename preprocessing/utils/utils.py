import json
import re

import numpy as np


def get_dataset_statistic(dataset):
    lens = np.array([len(element["labels"]) for element in dataset])

    need_to_predict_lens = np.array(
        [(np.array(element["labels"]) == -100).sum() for element in dataset]
    )

    not_need_to_predict_lens = np.array(
        [(np.array(element["labels"]) != -100).sum() for element in dataset]
    )

    print(f"Mean {np.mean(lens)}")
    print(f"Median {np.median(lens)}")
    print(f"Min {np.min(lens)}")
    print(f"Max {np.max(lens)}")
    print(f"STD {np.std(lens)}")

    first_sample = dataset[0]
    labels = first_sample["labels"]
    input_ids = first_sample["input_ids"]
    labels[labels == -100] = 0
    print("#####\noriginal: ", dataset.tokenizer.decode(input_ids), "\n#####")
    print("#####\nmasked: ", dataset.tokenizer.decode(labels), "\n#####")

    print(f"Need to predict total {np.sum(need_to_predict_lens)}")
    print(f"Need to predict avg {np.mean(need_to_predict_lens)}")

    print(f"Not need to predict total {np.sum(not_need_to_predict_lens)}")
    print(f"Not need to predict avg {np.mean(not_need_to_predict_lens)}")


def remove_links(text):
    link_reg = r"(https?:\/\/[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[\na-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    text = re.sub(link_reg, "", text)
    return text


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
