import os
import json


def convert_funsd_to_paddleocr(input_dir, output_file):
    """
    Converts FUNSD dataset annotations from JSON format to the format required by PaddleOCR.

    Parameters:
    - input_dir (str): Directory containing JSON files for each image.
    - output_file (str): Path to the output file where the converted annotations will be saved.
    """
    annotations = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(input_dir, filename)
            image_name = filename.replace('.json', '.png')

            with open(json_path, 'r', encoding="utf-8") as file:
                data = json.load(file)

                # Process each annotation in the JSON file
                paddle_ocr_annotations = []
                for annotation in data['form']:
                    flat_box = annotation['box']  # 'box' is [x1, y1, x2, y2]
                    # Convert flat box to a list of four points
                    points = [
                        [flat_box[0], flat_box[1]],  # Top-left
                        [flat_box[2], flat_box[1]],  # Top-right
                        [flat_box[2], flat_box[3]],  # Bottom-right
                        [flat_box[0], flat_box[3]]  # Bottom-left
                    ]
                    transcription = annotation['text'] if annotation['text'].strip() != "" else "###"

                    paddle_ocr_annotations.append({
                        'transcription': transcription,
                        'points': points
                    })

                annotations.append(f"{image_name}\t{json.dumps(paddle_ocr_annotations)}")

    with open(output_file, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')


if __name__ == "__main__":
    train_input_dir = 'dataset/train/annotations'
    train_output_file = 'dataset/train/annotations.txt'
    convert_funsd_to_paddleocr(train_input_dir, train_output_file)

    valid_input_dir = 'dataset/validation/annotations'
    valid_output_file = 'dataset/validation/annotations.txt'
    convert_funsd_to_paddleocr(valid_input_dir, valid_output_file)