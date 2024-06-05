import argparse
import subprocess
from paddleocr import PaddleOCR
from utils import inference_model_exists, export_inference_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--det_model_dir",
    type=str,
    required=True,
    help="Path to the detection model directory",
)
parser.add_argument(
    "--rec_model_dir",
    type=str,
    required=True,
    help="Path to the recognition model directory",
)
parser.add_argument(
    "--image_dir",
    type=str,
    required=True,
    help="Path to the directory containing images to be processed",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to the directory to save the inference results",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if not inference_model_exists(args.det_model_dir):
        print("Detection inference model wasn't found. Exporting finetuned model.")
        export_inference_model(args.det_model_dir)

    if not inference_model_exists(args.rec_model_dir):
        print("Recognition inference model wasn't found. Exporting finetuned model.")
        export_inference_model(args.rec_model_dir)

    command = [
        "python", "./PaddleOCR/tools/infer/predict_system.py",
        "--image_dir", args.image_dir,
        "--det_model_dir", args.det_model_dir,
        "--rec_model_dir", args.rec_model_dir,
        "--rec_char_dict_path", "./PaddleOCR/ppocr/utils/ppocr_keys_v1.txt",
        "--draw_img_save_dir", args.output_dir,
        "--use_angle_cls", "False",
        "--use_space_char", "True",
        "--use_mp", "False",
        # "--total_process_num", "4",
        "--use_gpu", "False",
        "--show_log", "False"
    ]
    subprocess.run(command, shell=True, check=True)
