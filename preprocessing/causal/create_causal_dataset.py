import logging
import os.path
from glob import glob

import datasets
import hydra
from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig
from preprocessing.causal.CausalDatasetMining import (
    CausalDataset,
    build_hf_dataset_from_list,
)
from preprocessing.utils.utils import read_jsonl, write_jsonl
from transformers import AutoTokenizer

from src.dataset.Dataset import ChatDataset
from src.utils.logger import get_pylogger

log = get_pylogger(__name__)
logging.basicConfig(level=logging.INFO)


def json_task(cfg: DictConfig):
    r"""
    Read all source of data, process them and save json file for each into result folder
    Args:
        cfg: configuration dict describe all process
    """
    log.info(f"Datasets %s", list(cfg["datasets"].keys()))
    json_output_path = os.path.join(cfg["output_path"], "causal_json")
    for data_source_name, data_source_preproc_config in cfg["datasets"].items():
        log.info("Instantiate %s", data_source_name)

        dataset: CausalDataset = hydra.utils.instantiate(data_source_preproc_config)
        json_data = dataset.run()

        full_json_path = os.path.join(json_output_path, f"{data_source_name}.jsonl")
        os.makedirs(json_output_path, exist_ok=True)
        write_jsonl(json_data, full_json_path)


def dataset_task(cfg: DictConfig):
    r"""
    Take json file for each datasource, build dataset from them
    Args:
        cfg: configuration dict describe all process
    """
    input_path = os.path.join(cfg["output_path"], "causal_json")

    output_path = os.path.join(cfg["output_path"], "causal")
    json_dataset_path = os.path.join(input_path, "*.jsonl")
    json_datasets = glob(json_dataset_path)
    final_datasets = []
    for json_dataset in json_datasets:
        data = read_jsonl(json_dataset)
        final_dataset = build_hf_dataset_from_list(
            template_dialog=data,
            tokenizer_name=cfg["tokenizer_name"],
            conversation=hydra.utils.instantiate(cfg["conversation"]),
            max_token_count=cfg["max_token_count"],
        )
        print(len(final_dataset["train"]))
        final_datasets.append(final_dataset)

    train_dataset = concatenate_datasets([ds["train"] for ds in final_datasets])

    test_dataset = concatenate_datasets([ds["test"] for ds in final_datasets])

    train_records = ChatDataset(
        original_records=list(train_dataset),
        conversation=hydra.utils.instantiate(cfg["conversation"]),
        tokenizer=AutoTokenizer.from_pretrained(cfg["tokenizer_name"]),
        max_tokens_count=cfg["max_token_count"],
        debug=False,
    )

    test_dataset = ChatDataset(
        original_records=list(test_dataset),
        conversation=hydra.utils.instantiate(cfg["conversation"]),
        tokenizer=AutoTokenizer.from_pretrained(cfg["tokenizer_name"]),
        max_tokens_count=cfg["max_token_count"],
        debug=False,
    )

    all_datasets_merged = DatasetDict(
        {
            "train": datasets.Dataset.from_list([sample for sample in train_records]),
            "test": datasets.Dataset.from_list([sample for sample in test_dataset]),
        }
    )

    all_datasets_merged.save_to_disk(output_path)
    print(len(all_datasets_merged["train"]))
    print(len(all_datasets_merged["test"]))
    tokenizer_name_wo_nick = cfg["tokenizer_name"].split("/")[1]

    all_datasets_merged.push_to_hub(
        f'{cfg["dataset_name"]}_{tokenizer_name_wo_nick}', private=True
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    r"""Dataset creation starting point.

    There are two main stage of building dataset:
    1) Create json files for each source
    2) Build dataset and push it to hugging face
    Args:
        cfg: configuration dict describe all process
    """
    if cfg["task"] == "json":
        json_task(cfg)
    elif cfg["task"] == "dataset":
        dataset_task(cfg)
    else:
        raise ValueError("Task should be json or dataset")


if __name__ == "__main__":
    main()
