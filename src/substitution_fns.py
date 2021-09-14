import copy
import gzip
import json
import random
import typing
from collections import Counter, defaultdict

from src.classes.answer import Answer
from src.classes.qadataset import QADataset
from src.classes.qaexample import QAExample
from src.utils import normalize_text


####################################################################################################
#########  SUBSTITUTION FUNCTIONS
####################################################################################################
def alias_substitution_fn(
    dset: QADataset,
    wikidata_info_path: str,
    replace_every: bool,
    max_aliases: int,
    category: str,
):
    """Derives a new dataset of modified examples, where the original answer has been replaced
    with one of it's own wikidata aliases.

    Args:
        dset: The original QADataset
        wikidata_info_path: A path to a mapping from wikidata ID to a dictionary of
            wikidata info (see extract_wikidata_info.py for details).
        replace_every: If True, replace every original answer in the substitution examples context, otherwise replace just the primary one.
        max_aliases: How many new (modified) examples to create from one original example. Each one
            will replace the original answer with one of it's aliases, up to min(max_aliases, the number of available aliases).
        category: This limits substitution generation to only use original examples with this answer type category. 
            `ALL` is an option.
    """

    def sub_fn(ex: QAExample):
        """Derive all modified examples from one original example."""
        # Determine which aliases are valid substitutions
        # Store these as Dict[alias_text --> GoldAnswer] so we can retrieve the info for each alias
        gold_answer_texts = [ga.text for ga in ex.gold_answers]
        alias_to_info = {
            alias: ga for ga in ex.gold_answers if ga.aliases for alias in ga.aliases
        }
        valid_aliases = list(set(alias_to_info.keys()) - set(gold_answer_texts))
        sub_exs = [
            # As the sub_answer is an alias we expect the metadata to be mostly the same as the original answer
            create_new_example(
                ex=ex,
                new_id=f"alias-sub-{idx}",
                answer_text=alias,
                ner_label=alias_to_info[alias].ner_label,
                kb_id=alias_to_info[alias].kb_id,
                wikidata_label=alias_to_info[alias].wikidata_label,
                aliases=[alias_to_info[alias].text],
                wikidata_types=alias_to_info[alias].wikidata_types,
                wikipedia_page=alias_to_info[alias].wikipedia_page,
                popularity=alias_to_info[alias].popularity,
                answer_type=alias_to_info[alias].answer_type,
                replace_every_original_answer=replace_every,
            )
            for idx, alias in enumerate(valid_aliases)
        ]
        return sub_exs

    new_exs, num_alias_dist = [], []
    for ex in dset.examples:
        ex_answer_type = ex.get_example_answer_type()
        if (
            category.lower() == "all"
            or (
                category.lower() == "nonnumeric"
                and category not in [None, "DATE", "NUMERIC"]
            )
            or category.lower() == ex_answer_type.lower()
        ):
            alias_exs = sub_fn(ex)
            # If not 0 then we select a subset of aliased substitution examples
            if max_aliases:
                alias_exs = alias_exs[:max_aliases]
            new_exs.extend(alias_exs)
            num_alias_dist.append(len(alias_exs))

    print(
        f"Num New Examples Generated per Original Example (using max-aliases={max_aliases}, category={category})): {Counter(num_alias_dist)}"
    )
    print(
        f"NB: The quantity of zeros reflects how many examples do not have wikidata IDs to draw aliases from."
    )
    print(f"Finished Alias Substitution.")
    return new_exs


def corpus_substitution_fn(
    dset: QADataset,
    wikidata_info_path: str,
    replace_every: bool,
    num_samples: int,
    category: str,
):
    """Derives a new dataset of modified examples, where the original answer has been replaced
    by another answer of the same `type` drawn randomly from the corpus of answers in the original dataset.
    This substitution function maintains the same distribution of answers as the original dataset.

    Args:
        dset: The original QADataset
        wikidata_info_path: A path to a mapping from wikidata ID to a dictionary of
            wikidata info (see extract_wikidata_info.py for details).
        replace_every: If True, replace every original answer in the substitution examples context, otherwise replace just the primary one.
        num_samples: How many new (modified) examples to create from one original example.
        category: This limits substitution generation to only use original examples with this answer type category. 
            `ALL` is an option.
    """
    # generate a corpus of substitute answers, keyed by answer type
    answer_corpus_by_groups = group_answers_by_answer_type(dset)

    def sub_fn(ex: QAExample):
        """Derive all modified examples from one original example."""
        new_exs = []
        ex_ans_typ = ex.get_example_answer_type()
        if ex_ans_typ is not None:
            for idx in range(num_samples):
                sub_answer = select_random_non_identical_answer(
                    ex, answer_corpus_by_groups[ex_ans_typ]
                )
                new_ex = create_new_example(
                    ex=ex,
                    new_id=f"corpus-sub-{idx}",
                    answer_text=sub_answer.text,
                    ner_label=sub_answer.ner_label,
                    kb_id=sub_answer.kb_id,
                    wikidata_label=sub_answer.wikidata_label,
                    aliases=sub_answer.aliases,
                    wikidata_types=sub_answer.wikidata_types,
                    wikipedia_page=sub_answer.wikipedia_page,
                    popularity=sub_answer.popularity,
                    answer_type=sub_answer.answer_type,
                    replace_every_original_answer=replace_every,
                )
                new_exs.append(new_ex)
        return new_exs

    new_exs = []
    for ex in dset.examples:
        ex_answer_type = ex.get_example_answer_type()
        if ex_answer_type is not None:
            if category.lower() == "all" or category.lower() == ex_answer_type.lower():
                exs = sub_fn(ex)
                new_exs.extend(exs)

    group_counter = Counter([ex.get_example_answer_type() for ex in new_exs])
    print(
        f"Num New Examples Generated by Answer Type Group (using num-samples={num_samples}, category={category}): {group_counter}"
    )
    print(
        f"NB: Not all original examples can be substituted, if their answer type is not discernable, or one of the 5 high-confidence identified by this NER model."
    )
    print(f"Finished Corpus Substitution.")
    return new_exs


def popularity_substitution_fn(
    dset: QADataset,
    wikidata_info_path: str,
    replace_every: bool,
    num_bins: int,
    max_ents_per_pop: int,
    category: str,
):
    """Derives a new dataset of modified examples, where the original answer has been replaced
    by a Wikidata entity of the same type, but with varying popularity. This
    substitution first splits all Wikidata entities into bins of equal sizes
    where each bin contains entities with similar popularities. For an original
    instance, we create a new substituted instance for each bin by sampling an entity from each bin.
    If `num_bins` is 1, then this function just samples from all of Wikidata.
    This substitution only operates on human answers since the notion of popularity
    for other answer types (e.g., dates) is ill-defined.

    Args:
        dset: The original QADataset
        wikidata_info_path: ``str`` A path to a mapping from wikidata ID to a
        dictionary of wikidata info (see extract_wikidata_info.py for details).
        replace_every: If True, replace every original answer in the substitution examples context, otherwise replace just the primary one.
        num_bins: ``int`` The number of bins which Wikidata entities are split into.
            Each bin contains an equal number of Wikidata entities group by
            popularity values. For each original instance, we create a substituted
            instance for each bin by sampling an entity from the bin.")
        max_ents_per_pop: ``int`` The number of entities to keep per popularity value
        category: This limits substitution generation to only use original examples with this 
            answer type category. `ALL` is an option. `PERSON` is the default for popularity
            substitution as it yields the most reliable values.
    """

    def sub_fn(ex: QAExample, wikidata_popularity_bins: typing.List[typing.Dict]):
        """Derive all modified examples from one original example."""
        new_exs = []

        for bin_id in range(len(wikidata_popularity_bins)):
            sub_qid = random.choice(list(wikidata_popularity_bins[bin_id].keys()))
            sub_qid_info = wikidata_popularity_bins[bin_id][sub_qid]

            new_ex = create_new_example(
                ex=ex,
                new_id=f"pop-sub-{bin_id}",
                answer_text=sub_qid_info["label"],
                ner_label=None,
                kb_id=sub_qid,
                wikidata_label=sub_qid_info["label"],
                aliases=sub_qid_info["aliases"],
                wikidata_types=sub_qid_info["entity_types"],
                wikipedia_page=sub_qid_info["wikipedia_page"],
                popularity=sub_qid_info["popularity"],
                answer_type=None,
                replace_every_original_answer=replace_every,
            )
            new_exs.append(new_ex)
        return new_exs

    wikidata_popularity_bins = bin_wikidata_entities_by_popularity(
        wikidata_info_path=wikidata_info_path,
        max_ents_per_pop=max_ents_per_pop,
        num_bins=num_bins,
    )

    new_exs = []
    for ex in dset.examples:
        ex_answer_type = ex.get_example_answer_type()
        if ex_answer_type is not None:
            if category.lower() == "all" or category.lower() == ex_answer_type.lower():
                exs = sub_fn(ex, wikidata_popularity_bins)
                new_exs.extend(exs)

    print(f"Finished Popularity Substitution, yielding {len(new_exs)} new examples.")
    return new_exs


def type_swap_substitution_fn(
    dset: QADataset,
    wikidata_info_path: str,
    replace_every: bool,
    num_samples: int,
    category: str,
):
    """Derives a new dataset of modified examples, where the original answer has been replaced
    by another answer of a different `type` drawn randomly from the corpus of answers in the original dataset.
    This substitution function is the same as corpus_substitution_fn except the answer types are different
    rather than the same.

    Args:
        dset: The original QADataset
        wikidata_info_path: A path to a mapping from wikidata ID to a dictionary of
            wikidata info (see extract_wikidata_info.py for details).
        replace_every: If True, replace every original answer in the substitution examples context, otherwise replace just the primary one.
        num_samples: How many new (modified) examples to create from one original example.
        category: This limits substitution generation to only use original examples with this answer type category. 
            `ALL` is an option.
    """
    # generate a corpus of substitute answers, keyed by answer type
    answer_corpus_by_groups = group_answers_by_answer_type(dset)
    group_types = list(answer_corpus_by_groups.keys())

    def sub_fn(ex: QAExample, target_group: str):
        """Derive all modified examples from one original example."""
        new_exs = []
        ex_ans_typ = ex.get_example_answer_type()
        if ex_ans_typ is not None:
            for idx in range(num_samples):
                sub_answer = select_random_non_identical_answer(
                    ex, answer_corpus_by_groups[target_group]
                )
                new_ex = create_new_example(
                    ex=ex,
                    new_id=f"type-swap-sub-{idx}",
                    answer_text=sub_answer.text,
                    ner_label=sub_answer.ner_label,
                    kb_id=sub_answer.kb_id,
                    wikidata_label=sub_answer.wikidata_label,
                    aliases=sub_answer.aliases,
                    wikidata_types=sub_answer.wikidata_types,
                    wikipedia_page=sub_answer.wikipedia_page,
                    popularity=sub_answer.popularity,
                    answer_type=sub_answer.answer_type,
                    replace_every_original_answer=replace_every,
                )
                new_exs.append(new_ex)
        return new_exs

    new_exs = []
    for ex in dset.examples:
        ex_answer_type = ex.get_example_answer_type()
        if ex_answer_type is not None:
            if category.lower() == "all" or category.lower() == ex_answer_type.lower():
                for target_group in group_types:
                    if target_group == ex.get_example_answer_type():
                        continue
                    exs = sub_fn(ex, target_group)
                    new_exs.extend(exs)

    group_counter = Counter(
        [
            (
                ex.get_example_answer_type(),
                ex.original_example.get_example_answer_type(),
            )
            for ex in new_exs
        ]
    )
    print(
        f"Num New Examples Generated by Answer Type Group (using num-samples={num_samples}, category={category})): {group_counter}"
    )
    print(
        f"NB: Not all original examples can be substituted, if their answer type is not discernable, or one of the 5 high-confidence identified by this NER model."
    )
    print(f"Finished Type Swap Substitution.")
    return new_exs


####################################################################################################
#########  SUBSTITUTION HELPER FUNCTIONS
####################################################################################################


def create_new_example(
    ex: QAExample,
    new_id: str,
    answer_text: str,
    ner_label: str,
    kb_id: str,
    wikidata_label: str,
    aliases: typing.List[str],
    wikidata_types: typing.List[str],
    wikipedia_page: str,
    popularity: int,
    answer_type: str,
    replace_every_original_answer: bool,
):
    """Creates a new example from the original example, given the specified new metadata. Copies the 
    original example, initializes a new answer, and applies the answer substitution to the example.
    
    Args:
        ex: The original example
        other args: See the Answer init function in src/classes/answer.py for full details.
        replace_every_original_answer: If False, only replace the main gold answer that appears
            in the text, otherwise replace all valid gold answers that appear in the text.
    """
    sub_ex = copy.deepcopy(ex)
    sub_answer = Answer(
        text=answer_text,
        spans=None,
        ner_label=ner_label,
        kb_id=kb_id,
        wikidata_label=wikidata_label,
        aliases=aliases,
        wikidata_types=wikidata_types,
        wikipedia_page=wikipedia_page,
        popularity=popularity,
        answer_type=answer_type,
    )
    sub_ex.apply_substitution(
        sub_answer,
        ex,
        new_id,
        replace_every_original_answer=replace_every_original_answer,
    )
    return sub_ex


def group_answers_by_answer_type(dset: QADataset):
    """Reorganizes a QADataset into a mapping from answer type to member answers."""
    group_to_answer_sets = defaultdict(dict)
    for ex in dset.examples:
        for answer in ex.gold_answers:
            if answer.answer_type:
                group_to_answer_sets[answer.answer_type][answer.text] = answer
    return group_to_answer_sets


def select_random_non_identical_answer(ex: QAExample, sample_set: typing.List[str]):
    """Randomly samples an answer from `sample_set` that is non-identical to the gold answers
    currently represented in the QAExample."""
    norm_gold_answers = {normalize_text(ga.text): ga for ga in ex.gold_answers}
    sample_keys = list(sample_set.keys())
    sub_key = None
    while not sub_key or normalize_text(sub_key) in norm_gold_answers:
        sub_key = random.choice(sample_keys)
    return sample_set[sub_key]


def bin_wikidata_entities_by_popularity(
    wikidata_info_path: str, num_bins: int, max_ents_per_pop: int = 10
):
    """Groups a list of Wikidata entities with the specified entity type into one of
    `num_bins` equally sized bins based on entitye popularity. We only include
    entities with the type "Q5" which is the Wikidata entity

    Args:
        wikidata_info_path: ``str`` Path to a .json.gz file containing Wikidata entity information
        num_bins: ``int`` The number of bins to do the grouping
        entity_type: ``str`` The entity type to do the grouping on. By default,
            we only include entities with the type "Q5" which is the Wikidata entity
        max_ents_per_pop: ``int`` The number of entities to keep per popularity value

    Returns:
        entity_popularity_bins: ``List[Dict]`` Returns a list where each list
        corresponds to a bin. Within each list if a dictionary with entity information.
    """
    with gzip.open(wikidata_info_path, "r") as reader:
        wikidata_info = json.load(reader)

    # Delete entities without the desired entity type
    for kb_id in list(wikidata_info.keys()):
        if "Q5" not in wikidata_info[kb_id]["entity_types"]:
            del wikidata_info[kb_id]

    # If there are multiple entities with the same pop, we only keep the first
    # `max_ents_per_pop`. This ensures that when we group entities into
    # equally sized bins, that we don't have too much bleeding where
    # entities with the same popularity are in different bins
    num_ents_per_pop = defaultdict(int)
    for kb_id in list(wikidata_info.keys()):
        pop = wikidata_info[kb_id]["popularity"]
        if num_ents_per_pop[pop] >= max_ents_per_pop:
            del wikidata_info[kb_id]
        num_ents_per_pop[pop] += 1

    # Sort entity IDs (QID) by their popularity
    # type(entities) = List[str]
    kb_ids = sorted(wikidata_info, key=lambda x: wikidata_info[x]["popularity"])

    # Split list of entities into a list (of len `num_bins`) of list of entities
    # type(entity_popularity_bins) = List[Dict]
    bin_size = len(kb_ids) // num_bins + 1
    entity_popularity_bins = [
        {kb_id: wikidata_info[kb_id] for kb_id in kb_ids[i : i + bin_size]}
        for i in range(0, len(kb_ids), bin_size)
    ]

    return entity_popularity_bins
