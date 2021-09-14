"""
Loads and preprocesses a dataset of your choice, saving it in a normalized format,
ready to apply downstream substitution functions.
"""
import argparse
import gzip
import json
import typing

import wget

from src.classes.qadataset import *
from src.utils import argparse_str2bool

# NB: Feel free to add custom datasets here.
DATASETS = {
    # MRQA Datasets available at: https://github.com/mrqa/MRQA-Shared-Task-2019
    "MRQANaturalQuestionsTrain": (
        MRQANaturalQuetsionsDataset,
        "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz",
    ),
    "MRQANaturalQuestionsDev": (
        MRQANaturalQuetsionsDataset,
        "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz",
    ),
}


def load_and_preprocess_dataset(args):
    dataset_class, url_or_path = DATASETS[args.dataset]
    dataset = dataset_class.new(args.dataset, url_or_path)
    dataset.preprocess(args.wikidata, args.ner_model, args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help=f"Name of the dataset. Must be one of {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "-w",
        "--wikidata",
        default="wikidata/entity_info.json.gz",
        help="Path to wikidata entity info file generated in Stage 2.",
    )
    parser.add_argument(
        "-m",
        "--ner-model",
        default="models/kc-ner-model/",
        help="Path to the directory of our SpaCy Named Entity Recognition and Entity Linking model, downloaded during setup.",
    )
    parser.add_argument(
        "--debug",
        type=argparse_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If set to True, only 100 examples are processed, to speed up debugging.",
    )
    args = parser.parse_args()
    load_and_preprocess_dataset(args)
