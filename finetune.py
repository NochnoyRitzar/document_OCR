import os
import yaml
import wandb
import argparse
import subprocess
from dotenv import load_dotenv


def update_wandb_config(config_path, wandb_project):
    """
    Update the WandB project name in the model configuration file.

    :param config_path: Path to the model configuration YAML file.
    :param wandb_project: WandB project name to be updated in the configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config.update({"wandb": {"project": wandb_project}})
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


def parse_args():
    """
    Parse command-line arguments for finetuning the PaddleOCR model.

    :return: Parsed arguments containing WandB project name and path to the model config YAML file.
    """
    parser = argparse.ArgumentParser(description="Finetune PaddleOCR model")
    parser.add_argument(
        "--wandb_project", type=str, required=True, help="WandB project name"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to model config YAML file"
    )
    return parser.parse_args()


def login_wandb():
    """
    Log in to WandB using the API key from environment variables.

    :raises ValueError: If the WANDB_API_KEY environment variable is not set.
    """
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("WANDB_API_KEY environment variable not set")
    wandb.login(key=wandb_key)


def start_finetuning(config_path, wandb_project):
    """
    Start the finetuning process for the PaddleOCR model.

    :param config_path: Path to the model configuration YAML file.
    :param wandb_project: WandB project name for tracking the finetuning process.
    """
    update_wandb_config(config_path, wandb_project)

    command = ["python", "./PaddleOCR/tools/train.py", "-c", config_path]
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    login_wandb()
    start_finetuning(args.config_path, args.wandb_project)
