import os
import json
import random
import shutil
from PIL import Image
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
            os.path.join(dataset_dir, "train", "annotations", f"{filename}.json"),
            os.path.join(validation_annotations_path, f"{filename}.json"),
        )

    print(f"Moved {len(validation_split)} pairs to validation set.")


def convert_funsd_to_paddleocr_format(input_dir, output_file):
    """
    Converts FUNSD dataset annotations from JSON format to the format required by PaddleOCR.

    Parameters:
    - input_dir (str): Directory containing JSON files for each image.
    - output_file (str): Path to the output file where the converted annotations will be saved.
    """
    annotations = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            image_name = filename.replace(".json", ".png")

            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                # Process each annotation in the JSON file
                paddle_ocr_annotations = []
                for annotation in data["form"]:
                    flat_box = annotation["box"]  # 'box' is [x1, y1, x2, y2]
                    # Convert flat box to a list of four points
                    points = [
                        [flat_box[0], flat_box[1]],  # Top-left
                        [flat_box[2], flat_box[1]],  # Top-right
                        [flat_box[2], flat_box[3]],  # Bottom-right
                        [flat_box[0], flat_box[3]],  # Bottom-left
                    ]
                    transcription = (
                        annotation["text"]
                        if annotation["text"].strip() != ""
                        else "###"
                    )

                    paddle_ocr_annotations.append(
                        {"transcription": transcription, "points": points}
                    )

                annotations.append(
                    f"{image_name}\t{json.dumps(paddle_ocr_annotations)}"
                )

    with open(output_file, "w") as f:
        for annotation in annotations:
            f.write(annotation + "\n")


def create_text_rec_dataset(dataset_path, split):
    output_path = os.path.join(dataset_path, split, "rec")
    annotations_output_file = os.path.join(dataset_path, split, "rec_gt_train.txt")
    os.makedirs(output_path, exist_ok=True)

    counter = 1
    with open(annotations_output_file, "w", encoding="utf-8") as ann_file:
        for annotation_file in os.listdir(
            os.path.join(dataset_path, split, "annotations")
        ):
            annotation_file_path = os.path.join(
                dataset_path, split, "annotations", annotation_file
            )
            image_file_name = annotation_file.replace(".json", ".png")
            image_file_path = os.path.join(
                dataset_path, split, "images", image_file_name
            )
            image = Image.open(image_file_path)

            # Open and parse the annotation JSON file
            with open(annotation_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            for text_instance in data["form"]:
                # Crop and save the image based on the box coordinates
                box = text_instance["box"]
                x1, y1, x2, y2 = box
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_image_file_name = (
                    f"{os.path.splitext(image_file_name)[0]}_{counter:03d}.png"
                )
                cropped_image_file_path = os.path.join(
                    output_path, cropped_image_file_name
                )
                cropped_image.save(cropped_image_file_path)

                ann_file.write(f"{cropped_image_file_path}\t{text_instance['text']}\n")
                counter += 1

    print(f"Successfully created text recognition {split} split")


if __name__ == "__main__":
    dataset_dir = "dataset"
    create_valid_split(dataset_dir)

    train_input_dir = os.path.join(dataset_dir, "train", "annotations")
    train_output_file = os.path.join(dataset_dir, "train", "annotations.txt")
    convert_funsd_to_paddleocr_format(train_input_dir, train_output_file)

    valid_input_dir = os.path.join(dataset_dir, "validation", "annotations")
    valid_output_file = os.path.join(dataset_dir, "validation", "annotations.txt")
    convert_funsd_to_paddleocr_format(valid_input_dir, valid_output_file)

    create_text_rec_dataset(dataset_dir, split="train")
    create_text_rec_dataset(dataset_dir, split="validation")

    print("Data preparation completed.")
