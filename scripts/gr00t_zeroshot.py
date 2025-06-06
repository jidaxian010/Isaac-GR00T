import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import json
import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.policy import Gr00tPolicy
from gr00t.utils.peft import get_lora_model
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError


@dataclass
class Config:
    """Configuration for GR00T model zero-shot inference."""

    # Dataset parameters
    dataset_path: str
    """Path to the dataset directory."""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "so100"  # Changed to so100 for SO100 dataset
    """Data configuration name from DATA_CONFIG_MAP."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1-2B"
    """Path or HuggingFace model ID for the base model."""

    # Data loading parameters
    embodiment_tag: str = "new_embodiment"  # Changed to new_embodiment for SO100
    """Embodiment tag to use for inference."""

    video_backend: str = "decord"
    """Video backend to use for inference."""

    metadata_path: str = "metadata.json"
    """Path to the metadata file."""




def main(config: Config):
    """Main inference function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
    )

    # check if dataset and metadata are valid
    

    # ------------ step 2: load model ------------
    model = GR00T_N1.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=False,  # No tuning for zero-shot
        tune_visual=False,  # No tuning for zero-shot
        tune_projector=False,  # No tuning for zero-shot
        tune_diffusion_model=False,  # No tuning for zero-shot
    )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"


    # ------------ step 4: save metadata ------------

    # 4.1 get metadata path
    try:
        # NOTE(YL) this returns the local path to the model which is normally
        # saved in ~/.cache/huggingface/hub/
        model_path = snapshot_download(config.base_model_path, repo_type="model")
        # HFValidationError, RepositoryNotFoundError
    except (HFValidationError, RepositoryNotFoundError):
        print(
            f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
        )
    metadata_path = Path(model_path) / "experiment_cfg" / "metadata.json"


    # # metadata path from gr00t_finetune.py (sus?)
    # output_dir = Path(config.output_dir)
    # exp_cfg_dir = output_dir / "experiment_cfg"
    # exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    # metadata_path = exp_cfg_dir / "metadata.json"


    print(f"make sure the metadata.json file is present at {metadata_path}")




    # Save metadata for the new embodiment
    metadata_json = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata_json = json.load(f)
    metadata_json.update(
        {dataset.tag: dataset.metadata.model_dump(mode="json")}
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f, indent=4)

    print(f"Model and metadata saved to {metadata_path}")
    print("You can now use this model for zero-shot inference on SO100 dataset")
    print(dataset.metadata)


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(Config)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T ZERO-SHOT INFERENCE CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    main(config)
