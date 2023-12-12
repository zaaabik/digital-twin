import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, Trainer

OmegaConf.register_new_resolver("div", lambda x, y: x // y)
OmegaConf.register_new_resolver("mult", lambda x, y: x * y)


def train(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg["dataset"])

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    model: AutoModelForCausalLM = hydra.utils.instantiate(
        cfg["model"]["architecture"],
        torch_dtype=hydra.utils.get_object(cfg["model"]["architecture"]["torch_dtype"]),
    )
    model.requires_grad_(True)

    trainer: Trainer = hydra.utils.instantiate(
        cfg["trainer"],
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        _convert_="partial",
    )

    trainer.train()


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
