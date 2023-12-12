import logging
import os
from glob import glob

import hydra
from datasets import Dataset, concatenate_datasets
from omegaconf import DictConfig
from preprocessing.retrieval.src.Retrieval import RetrievalMining
from preprocessing.utils.utils import read_jsonl

from src.utils.logger import get_pylogger

log = get_pylogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Create dataset for qa, and push it to hub
    Args:
        cfg: configuration file
    """
    log.info(f"Datasets %s", list(cfg["datasets"].keys()))
    for key, dataset_config in cfg["datasets"].items():
        log.info(f"Instantiate %s", key)
        dataset: RetrievalMining = hydra.utils.instantiate(dataset_config)
        dataset.run()
        dataset.save_results()

    datasets = [
        Dataset.from_list(read_jsonl(dataset_path))
        for dataset_path in glob(os.path.join(cfg["output_path"], "*.jsonl"))
    ]
    dataset = concatenate_datasets(datasets)
    dataset.push_to_hub(cfg["dataset_name"], private=True)


if __name__ == "__main__":
    main()
