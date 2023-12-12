import logging

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from preprocessing.retrieval.src.Retrieval import FaissIndexCreator

from src.utils.logger import get_pylogger

log = get_pylogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base="1.3", config_path="configs", config_name="config_faiss.yaml")
def main(cfg: DictConfig):
    """
    Create faiss index from dataset
    Args:
        cfg: configuration file
    """
    dataset = load_dataset(cfg["qa_input_dataset"])["train"]
    processor: FaissIndexCreator = hydra.utils.instantiate(cfg["processor"])
    qa_faiss_dataset = processor.prepare(dataset)
    model_name_wo_username = cfg["processor"]["model_name"].split("/")[1]
    qa_faiss_dataset.save_faiss_index(
        "embeddings", cfg["faiss_index_output"] + f"_{model_name_wo_username}"
    )


if __name__ == "__main__":
    main()
