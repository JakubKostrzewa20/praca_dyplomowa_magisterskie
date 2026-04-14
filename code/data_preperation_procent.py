import os
import random
import shutil
from pathlib import Path

INPUT_DIR = "input_data/color"
OUTPUT_DIR = "input_data/new_datasets/100"
PERCENT_USE = 1
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1

input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)

folder_names = ["train", "val", "test"]

for class_dir in input_path.iterdir():
    if not class_dir.is_dir:
        continue

    class_name = class_dir.name
    print(class_name, "being proccesed")
    images = list(class_dir.glob("*"))

    num_to_use = int(len(images) * PERCENT_USE)
    images = images[:num_to_use]

    train_num = int(len(images) * TRAIN_RATIO)
    val_num = train_num + int(len(images) * VAL_RATIO)

    train_images = images[:train_num]
    val_images = images[train_num:val_num]
    test_images = images[val_num:]

    split = {"train": train_images, "val": val_images, "test": test_images}

    for split_name, image_list in split.items():
        target_dir = output_path / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_list:
            shutil.copy(image_path, target_dir)
