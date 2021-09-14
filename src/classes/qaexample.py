import json
import re
import typing
from collections import Counter

from src.classes.answer import Answer


class QAExample(object):
    """A Question Answering Example."""

    def __init__(
        self,
        uid: str,
        query: str,
        context: str,
        gold_answers: typing.List[Answer],
        is_substitute: bool = False,
        metadata: typing.Dict[str, str] = None,
        original_example: "QAExample" = None,
    ):
        """
        Do not invoke directly. Use `new` or `json_load`.

        Fields:
            uid: A unique identifier for this example.
            query: The query. 
            context: The context passage.
            gold_answers: A list of `Answer` objects, that represent gold answers to the question.
            is_substitute: Whether or not this example is the original or a substitute.
            metadata: Any additional info the user can populate here.
            original_example: If this example is a substitute, then this field oints to the 
                full original example.
        """
        self.uid = uid
        self.query = query
        self.context = context
        self.gold_answers = gold_answers
        self.metadata = metadata
        self.is_substitute = is_substitute
        self.original_example = original_example

    @classmethod
    def new(
        cls,
        uid: str,
        query: str,
        context: str,
        answers: typing.List[str],
        is_substitute: bool = False,
        metadata: typing.Dict[str, str] = None,
    ):
        """Instantiates and returns a new QAExample.

        Given textual answers, it finds them in the context and instantiates them 
        as `Answer` objects.
        """
        gold_answers = []
        for text in answers:
            context_spans = cls._find_answer_in_context(text, context)
            gold_answers.append(Answer(text, spans=context_spans))
        return cls(
            uid=uid,
            query=query,
            context=context,
            gold_answers=gold_answers,
            is_substitute=is_substitute,
            metadata=metadata,
            original_example=None,
        )

    @classmethod
    def json_load(cls, json_obj):
        """Loads a json dump of a QAExample. Call this after `self.json_dump`."""
        obj = json.loads(json_obj)
        return cls(
            uid=obj["uid"],
            query=obj["query"],
            context=obj["context"],
            gold_answers=[Answer.json_load(ga_obj) for ga_obj in obj["gold_answers"]],
            is_substitute=obj["is_substitute"],
            metadata=obj["metadata"],
            original_example=obj["original_example"],
        )

    @classmethod
    def _find_answer_in_context(cls, answer_text: str, context: str):
        """Finds all instances of the `answer_text` in the context passage.
        
        Returns a list of (start index, end index) tuples.
        """
        context_spans = [
            (m.start(), m.end())
            for m in re.finditer(re.escape(answer_text.lower()), context.lower())
        ]
        return context_spans

    def json_dump(self, save_full: bool = False):
        """Creates a json dump of this QAExample.
        
        save_full: whether to save all of the original example, or just it's `uid`.
        """
        save_obj = {
            "uid": self.uid,
            "query": self.query,
            "context": self.context,
            "metadata": self.metadata,
            "is_substitute": self.is_substitute,
            "gold_answers": [ga.json_dump() for ga in self.gold_answers],
            "original_example": None,
        }
        if self.original_example:
            if save_full:
                save_obj["original_example"] = self.original_example.json_dump()
            else:
                save_obj["original_example"] = self.original_example.uid
        return save_obj

    def apply_substitution(
        self,
        sub_answer: Answer,
        original_example: "QAExample",
        sub_type: str,
        replace_every_original_answer: bool = False,
    ):
        """Applies the substitution to this example, modifying it's own uid, context 
        and gold answers.

        Args:
            sub_answer: The new Answer object that is replacing the existing gold_answers
            original_example: A copy of the original example to be saved
            sub_type: a prefix that represents this type of substitution, saved as part of the new
                uid.
            replace_every_original_answer: If False, only replace the main gold answer that appears
                in the text, otherwise replace all valid gold answers that appear in the text.
        """
        # update uid
        self.uid = f"{sub_type}_{self.uid}"
        # update context
        self.update_context_with_substitution(
            sub_answer, replace_every_original_answer=replace_every_original_answer
        )
        # update new answer with spans
        spans = self._find_answer_in_context(sub_answer.text, self.context)
        self.gold_answers = [sub_answer]
        self.is_substitute = True
        self.original_example = original_example

    def update_context_with_substitution(
        self, sub_answer: Answer, replace_every_original_answer=False
    ):
        """Replace all found instances of the answer in the context."""
        replace_spans = []
        replace_answers = (
            self.gold_answers
            if replace_every_original_answer
            else [a for a in self.gold_answers if a.is_answer_in_context()][0]
        )
        for orig_answer in self.gold_answers:
            replace_spans.extend(
                self._find_answer_in_context(orig_answer.text, self.context)
            )
        # Find and replace all string variants that correspond to the original answer in the context
        replace_strs = set([self.context[span[0] : span[1]] for span in replace_spans])
        for replace_str in replace_strs:
            self.context = self.context.replace(replace_str, sub_answer.text)

    def get_answers_in_context(self):
        """Find all gold answers that appear in the context passage, excluding those that don't."""
        return [ga for ga in self.gold_answers if ga.is_answer_in_context()]

    def get_example_answer_type(self):
        """For an example with multiple answers (potentially of different types) this function
        decides what type of answer the query likely needs for substitutions.
        """
        answer_type_counts = Counter(
            [answer.answer_type for answer in self.gold_answers if answer.answer_type]
        )
        most_common_types = answer_type_counts.most_common(1)
        if most_common_types:
            return most_common_types[0][0]
        return None

    def __repr__(self):
        return f"{self.uid} | {self.query} | {self.context[:100]} ..."
