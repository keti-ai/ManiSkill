# from pathlib import Path
# import os
# DEMO_DIR = Path(
#     os.getenv("MS_ASSET_DIR", os.path.join(os.path.expanduser("~"), ".maniskill/demos"))
# )
# pass
#
from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
dataset = ManiSkillTrajectoryDataset(dataset_file="/home/syh/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5")
data = dataset[150]
for k, v in data.items():
    print(k, v)