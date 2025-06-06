import os
import torch
import gr00t
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
MODEL_PATH = "./checkpoints/libero-object-checkpoints"
# MODEL_PATH = "./checkpoints/so100-checkpoints"
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/libero_object_data")
# DATASET_PATH = os.path.join(REPO_PATH, "demo_data/so100_strawberry_grape")
EMBODIMENT_TAG = "libero_arm"
# EMBODIMENT_TAG = "so100"

device = "cuda" if torch.cuda.is_available() else "cpu"
data_config = DATA_CONFIG_MAP["custom_panda_hand"]
# data_config = DATA_CONFIG_MAP["so100"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()


## load the policy
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

print("\n=== Policy Configuration ===")
print("Model:", policy.model)
print("\nModality Config Keys:", modality_config.keys())

print("\n=== Modality Config Details ===")
for key, value in modality_config.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: shape={value.shape}")
    else:
        print(f"{key}: {value}")


## Load the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="torchvision_av",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)

step_data = dataset[0]
print("\n=== Step Data Contents ===")
print("Keys:", step_data.keys())
for key, value in step_data.items():
    if isinstance(value, np.ndarray):
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {value.dtype}")
        print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
        if value.size < 10:  # Only print small arrays
            print(f"  Values: {value}")
    else:
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, list) and len(value) < 10:
            print(f"  Values: {value}")

print("\n=== Model Input Requirements ===")
print("The GR00T model expects input with the following keys:")
print("1. Video data (e.g., 'video.agentview_rgb'):")
print("   - Shape: (T, H, W, C) where T=1 for single frame")
print("   - Type: uint8")
print("   - Range: [0, 255]")
print("\n2. State data (e.g., 'state.single_arm', 'state.gripper'):")
print("   - Shape: (T, D) where T=1 for single timestep")
print("   - Type: float32")
print("   - Contains robot state information")
print("\n3. Language data (e.g., 'annotation.human.task_description'):")
print("   - Type: str or int")
print("   - Contains task description or instruction")

predicted_action = policy.get_action(step_data)
print("\n=== Predicted Action ===")
for key, value in predicted_action.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
    else:
        print(f"{key}: {type(value)}")
