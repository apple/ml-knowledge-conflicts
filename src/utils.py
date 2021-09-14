import os
import re
import string
import time
import typing

import spacy
from tqdm import tqdm


def argparse_str2bool(v):
    """Infers whether an argparse input indicates True or False."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def run_ner_linking(texts: typing.List[str], ner_model_path: str):
    """Loads and runs the Named Entity Recognition + Entity Linking model on all `texts`, 
    saving their named entity labels and Wikidata IDs if found.
    """
    nlp = spacy.load(ner_model_path)

    text_to_info = {}
    for text in tqdm(texts, desc="Running Named Entity Linking"):
        doc = nlp(text)
        datum = []
        for e in doc.ents:
            kb_id = (
                e.kb_id_ if "Q" == e.kb_id_[0] and e.kb_id_[1:].isnumeric() else None
            )
            datum.append(
                {"text": e.text, "label": e.label_, "id": kb_id,}
            )
        text_to_info[text] = datum
    return text_to_info


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class BasicTimer(object):
    """A basic timer that computes elapsed time of linear intervals."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._running = True
        self._total = 0.0
        self._start = round(time.time(), 2)
        self._interval_time = round(time.time(), 2)
        print(f"Timer [{self._name}] starting now")

    def reset(self) -> "BasicTimer":
        self._running = True
        self._total = 0
        self._start = round(time.time(), 2)
        return self

    def interval(self, intervalName: str):
        intervalTime = self._to_hms(round(time.time() - self._interval_time, 2))
        print(f"Timer [{self._name}] interval [{intervalName}]: {intervalTime}")
        self._interval_time = round(time.time(), 2)
        return intervalTime

    def stop(self) -> "BasicTimer":
        if self._running:
            self._running = False
            self._total += round(time.time() - self._start, 2)
        return self

    def resume(self) -> "BasicTimer":
        if not self._running:
            self._running = True
            self._start = round(time.time(), 2)
        return self

    def time(self) -> float:
        if self._running:
            return round(self._total + time.time() - self._start, 2)
        return self._total

    def finish(self) -> None:
        if self._running:
            self._running = False
            self._total += round(time.time() - self._start, 2)
            elapsed = self._to_hms(self._total)
        print(f"Timer [{self._name}] finished in {elapsed}")

    def _to_hms(self, seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%dh %02dm %02ds" % (h, m, s)
