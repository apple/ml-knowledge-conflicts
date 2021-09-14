import typing


class Answer(object):
    """An Answer (in a QAExample) with all relevant metadata."""

    def __init__(
        self,
        text: str,
        spans: typing.List[typing.Tuple[int, int]] = None,
        ner_label: str = None,
        kb_id: str = None,
        wikidata_label: str = None,
        aliases: typing.List[str] = None,
        wikidata_types: typing.List[str] = None,
        wikipedia_page: str = None,
        popularity: str = None,
        answer_type: str = None,
    ):
        """
        Fields:
            text: The raw, unchanged answer text.
            spans: The start and end character spans of this Answer in the context 
                passage of it's QAExample.
            ner_label: The Named Entity category label given to this Answer by the NER model.
                Will often be `None` if no category assigned.
            kb_id: The knowledge base ID, in this case Wikidata ID found by 
                the NER/Entity Linker model. Will often be `None` if not found.
            wikidata_label: The official Wikidata name corresponding to the found Wikidata ID.
            aliases: The official Wikidata aliases for the text associated with this Wikidata ID.
            wikidata_types: List of Wikidata entity types associated with this Wikidata ID.
            wikipedia_page: The Wikipedia page associated with this Wikidata ID.
            popularity: The popularity, measured in daily page views, associated with this 
                Wikidata ID.
            answer_type: The answer type derived by `_select_answer_type`, using NER and Wikidata
                metadata. This field is used to ensure some substitutions are "type-preserving",
                i.e. the chosen substitute answer is a coherent replacement in the context.
        """
        self.text = text
        self.spans = spans
        self.ner_label = ner_label
        self.kb_id = kb_id

        # Fields supplied by Wikidata Entity Info (from Stage 2) if self.kb_id identified.
        self.wikidata_label = wikidata_label
        self.aliases = aliases
        self.wikidata_types = wikidata_types
        self.wikipedia_page = wikipedia_page
        self.popularity = popularity

        # Derived from `ner_label` and `wikidata_types`
        self.answer_type = answer_type

    def update_ner_info(self, ner_label: str, kb_id: str):
        """Updates the Answer fields with info found by the NER+EL model."""
        self.ner_label = ner_label
        self.kb_id = kb_id

    def update_wikidata_info(
        self,
        label: str = None,
        aliases: typing.List[str] = None,
        entity_types: typing.List[str] = None,
        wikipedia_page: str = None,
        popularity: int = None,
    ):
        """Updates the Answer fields with Wikidata entity info, if there is a Wikidata ID."""
        self.wikidata_label = label
        self.aliases = aliases
        self.wikidata_types = [etype.lower() for etype in entity_types]
        self.wikipedia_page = wikipedia_page
        self.popularity = popularity

        self.answer_type = self._select_answer_type()

    def is_equivalent(self, t: str):
        """Checks if this Answer is textually equivalent to some other string."""
        if t == self.text:
            return True
        if t.lower() == self.text.lower():
            return True
        return False

    def is_answer_in_context(self):
        """Checks if this Answer appears in the context."""
        if self.spans:
            return True
        return False

    def _select_answer_type(self):
        """Assigns this Answer a category/type. This type is used by substitution functions to 
        ensure the resulting substitution is coherent, or type-preserving.

        NB: This function can be edited by the user if they have their own NER model, or another
            set of answer types / substitutions they wish to analyze.
        """
        answer_type = None
        if self.ner_label == "PERSON":
            answer_type = "PERSON"
        elif self.ner_label == "DATE" or (
            self.ner_label == "CARDINAL" and "year" in self.wikidata_types
        ):
            answer_type = "DATE"
        elif self.ner_label in ["CARDINAL", "QUANTITY"]:
            answer_type = "NUMERIC"
        elif self.ner_label in ["GPE", "LOC"]:
            answer_type = "LOCATION"
        elif self.ner_label == "ORG":
            answer_type = "ORGANIZATION"
        return answer_type

    def json_dump(self):
        return self.__dict__

    @classmethod
    def json_load(cls, obj: typing.Dict[str, typing.Any]):
        """Loads this object from a Json Dict."""
        return cls(**obj)

    def __repr__(self):
        return f'Text: "{self.text}" | Spans: {self.spans} | Label: {self.ner_label} | WikidataID: {self.kb_id}'
