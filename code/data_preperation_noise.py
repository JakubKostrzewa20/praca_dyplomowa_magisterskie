import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageFilter


INPUT_DIR = "input_data/color"
OUTPUT_DIR = "input_data/new_datasets/noise/25"
PERCENT_NOISE = 0.25 
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

    train_num = int(len(images) * TRAIN_RATIO)
    val_num = train_num + int(len(images) * VAL_RATIO)

    train_images = images[:train_num]
    val_images = images[train_num:val_num]
    test_images = images[val_num:]

    num_to_blur = int(len(train_images)*PERCENT_NOISE)
    blur_id = set(random.sample(range(len(train_images)), num_to_blur))

    split = {"train": train_images, "val": val_images, "test": test_images}

    for split_name, image_list in split.items():
        target_dir = output_path / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for idx, image_path in enumerate(image_list):
            target_path = target_dir / image_path.name
            if split_name == "train" and idx in blur_id:
                img = Image.open(image_path)
                img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(1,7)))
                img.save(target_path)
            else:
                shutil.copy(image_path, target_dir)
