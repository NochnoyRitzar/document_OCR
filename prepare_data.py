import os
import random
import shutil
from utils import get_split_files
random.seed(42)


def create_valid_split(dataset_dir, split_ratio=0.2):
    validation_images_path = os.path.join(dataset_dir, "validation", "images")
    validation_annotations_path = os.path.join(dataset_dir, "validation", "annotations")
    os.makedirs(validation_images_path, exist_ok=True)
    os.makedirs(validation_annotations_path, exist_ok=True)

    image_files, annotation_files = get_split_files(dataset_dir, "train")

    base_names = [os.path.splitext(f)[0] for f in image_files]
    random.shuffle(base_names)
    split_point = int(len(base_names) * split_ratio)
    validation_split = base_names[:split_point]


    for filename in validation_split:
        shutil.move(
            os.path.join(dataset_dir, "train", "images", f"{filename}.png"),
            os.path.join(validation_images_path, f"{filename}.png"),
        )
        shutil.move(
            os.path.join(dataset_dir, "train",  "annotations", f"{filename}.json"),
            os.path.join(validation_annotations_path, f"{filename}.json"),
        )

    print(f"Moved {len(validation_split)} pairs to validation set.")


if __name__ == "__main__":
    dataset_dir = "dataset"
    create_valid_split(dataset_dir)
