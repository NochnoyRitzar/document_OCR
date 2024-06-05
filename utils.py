import os
import pandas as pd


def get_split_files(data_dir: str, data_split: str) -> tuple[list[str], list[str]]:
    """
    Get the list of image and annotation files for a given split
    :param data_dir: path to the dataset directory
    :param data_split: dataset split (train, validation, test)
    :return:
    """

    images_path = os.path.join(data_dir, data_split, "images")
    annotations_path = os.path.join(data_dir, data_split, "annotations")

    image_files = [f for f in os.listdir(images_path) if f.endswith(".png")]
    annotation_files = [f for f in os.listdir(annotations_path) if f.endswith(".json")]

    return image_files, annotation_files


def create_df_from_txt(txt_path, split) -> pd.DataFrame:
    """
    Create Dataframe from text file

    :return:
    """
    df = pd.read_csv(
        txt_path,
        sep="\t",
        names=["origin", "text"],
        header=None,
        encoding="utf-8",
    )
    df = df.dropna()
    df["text_len"] = df["text"].apply(lambda x: len(x))
    df["data_split"] = split

    return df


def replace_chars(text):
    text = text.replace("–", "-")
    text = text.replace("ο", "o")
    return text
