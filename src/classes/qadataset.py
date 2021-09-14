import gzip
import json
import os
import typing
from collections import defaultdict

import wget

from src.classes.qaexample import QAExample
from src.utils import BasicTimer, run_ner_linking

ORIG_DATA_DIR = "datasets/original/"
NORM_DATA_DIR = "datasets/normalized/"


class QADataset(object):
    """
    The base class for Question Answering Datasets that are prepared for 
    substitution functions.
    """

    def __init__(
        self,
        name: str,
        original_path: str,
        preprocessed_path: str,
        examples: typing.List[QAExample] = None,
    ):
        """Do not invoke directly. Use `new` or `load`.
        
        Fields:
            name: The name of the dataset --- also used to derive the save path.
            original_path: The original path of the unprocessed data.
            preprocessed_path: The path to the data after processing and saving.
            examples: A list of QAExamples in this dataset. This field is populated by
                `self.read_original_dataset` and later augmented by `self.preprocess`.
        """
        self.name = name
        self.original_path = original_path
        self.preprocessed_path = preprocessed_path
        self.examples = examples

    @classmethod
    def new(cls, name: str, url_or_path: str):
        """Returns a new QADataset object.

        Args:
            name: Identifying name of this dataset.
            url_or_path: Either the URL to download from, or the local path to read from.
        """
        if os.path.exists(url_or_path):
            original_path = url_or_path
        else:
            file_suffix = ".".join(os.path.basename(url_or_path).split(".")[1:])
            original_path = os.path.join(ORIG_DATA_DIR, f"{name}.{file_suffix}")
            cls._download(name, url_or_path, original_path)
        preprocessed_path = cls._get_norm_dataset_path(name)
        return cls(name, original_path, preprocessed_path)

    @classmethod
    def load(cls, name: str):
        """Loads and returns a QADataset object that has already been 
        `self.preprocess` and `self.save()`d

        Args:
            name: Identifying name of this dataset.
        """
        preprocessed_path = cls._get_norm_dataset_path(name)
        assert os.path.exists(
            preprocessed_path
        ), f"Preprocessed dataset should be at {preprocessed_path}."
        with gzip.open(preprocessed_path, "r") as inf:
            header = json.loads(inf.readline())
            assert header["dataset"] == name
            examples = [QAExample.json_load(l) for l in inf.readlines()]

        print(f"Read {len(examples)} examples from {preprocessed_path}")
        return cls(name, header["original_path"], preprocessed_path, examples)

    @classmethod
    def _get_norm_dataset_path(self, name: str):
        """Formats the path to the normalized/preprocessed data."""
        return os.path.join(NORM_DATA_DIR, f"{name}.jsonl.gz")

    @classmethod
    def _download(cls, name: str, url: str, dest_path: str):
        """Downloads the original dataset from `url` to `dest_path`."""
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            print(f"Downloading Original Dataset: {name}")
            wget.download(url, dest_path)

    def save(self):
        """Save the preprocessed dataset to JSONL.GZ file. Can be loaded using `self.load()`."""
        os.makedirs(os.path.dirname(self.preprocessed_path), exist_ok=True)
        with gzip.open(self.preprocessed_path, "wt") as outf:
            json.dump({"dataset": self.name, "original_path": self.original_path}, outf)
            outf.write("\n")
            for ex in self.examples:
                json.dump(ex.json_dump(), outf)
                outf.write("\n")
        print(f"Saved preprocessed dataset to {self.preprocessed_path}")

    def read_original_dataset(self, file_path: str):
        """Reads the original/raw dataset into a List of QAExamples.
        
        NB: This is to be implemented by QADataset subclasses, for the specific 
        dataset they represent.
        """
        pass

    def preprocess(
        self, wikidata_info_path: str, ner_model_path: str, debug: bool = False
    ):
        """Read the original dataset, normalize its format and preprocess it. This includes
        running the NER model on the answers, and linking those to wikidata for additional
        metadata that can be used in the logic of answer subsitution functions.

        Args:
            wikidata_info_path: Path to the wikidata entity info saved from Step 1.
            ner_model_path: Path to our SpaCy NER model, downloaded during setup.
            debug: If true, only sample 500 examples to quickly check everything runs end-t-end.
        """
        timer = BasicTimer(f"{self.name} Preprocessing")
        examples = self.read_original_dataset(self.original_path)
        if debug:  # Look at just a subset of examples if debugging
            examples = examples[:500]
        print(f"Processing {len(examples)} Examples...")

        self.label_entities(examples, ner_model_path)
        timer.interval("Labelling and Linking Named Entities")
        self.wikidata_linking(examples, wikidata_info_path)
        timer.interval("Wikidata and Popularity Linking")
        self.examples = examples

        self._report_dataset_stats()
        self.save()
        timer.finish()

    def label_entities(self, examples: typing.List[QAExample], ner_model_path: str):
        """Populate each answer with the NER labels and wikidata ID, if found."""
        all_answers = [answer.text for ex in examples for answer in ex.gold_answers]
        answers_to_info = run_ner_linking(all_answers, ner_model_path)

        for ex in examples:
            for answer in ex.gold_answers:
                # for each match found within the answer
                for ner_info in answers_to_info[answer.text]:
                    if answer.is_equivalent(ner_info["text"]):
                        answer.update_ner_info(
                            ner_info["label"], ner_info["id"]
                        )  # update answer

    def wikidata_linking(
        self, examples: typing.List[QAExample], wikidata_info_path: str
    ):
        """Using the answer's wikidata IDs (if found), extracts wikidata metadata."""
        with gzip.open(wikidata_info_path, "r") as inf:
            wikidata_info = json.load(inf)

        for ex in examples:
            for answer in ex.gold_answers:
                if answer.kb_id in wikidata_info:
                    answer.update_wikidata_info(**wikidata_info[answer.kb_id])

    def _report_dataset_stats(self):
        """Reports basic statistics on what is contained in a preprocessed dataset."""
        grouped_examples = defaultdict(list)
        for ex in self.examples:
            grouped_examples[ex.get_example_answer_type()].append(ex)

        print("Dataset Statistics")
        print("-------------------------------------------")
        print(f"Total Examples = {len(self.examples)}")
        for group, ex_list in grouped_examples.items():
            print(f"Answer Type: {group} | Size of Group: {len(ex_list)}")
        print("-------------------------------------------")


class MRQANaturalQuetsionsDataset(QADataset):
    """The QADatast for MRQA-Natural Questions.
    
    Original found here: https://github.com/mrqa/MRQA-Shared-Task-2019
    """

    def read_original_dataset(self, file_path: str):
        """Reads the original/raw dataset into a List of QAExamples.
        
        Args:
            file_path: Local path to the dataset.

        Returns:
            List[QAExample]
        """
        examples = []
        with gzip.open(file_path, "rb") as file_handle:
            header = json.loads(file_handle.readline())["header"]
            for entry in file_handle:
                entry = json.loads(entry)
                for qa in entry["qas"]:
                    examples.append(
                        QAExample.new(
                            uid=qa["qid"],
                            query=qa["question"],
                            context=entry["context"],
                            answers=qa["answers"],
                            metadata={},  # NB: Put any metadata you wish saved here.
                        )
                    )
        return examples
