"""
Given the substitution function of your choice, will derive a new dataset out of the preprocessed
one you specify, applying the appropriate substitution rules.
"""
import argparse
import copy
import gzip
import inspect
import json
import os
import typing

from src.classes.qadataset import QADataset
from src.substitution_fns import *
from src.utils import argparse_str2bool

""" NB: Feel free to add custom functions here. """
SUBSTITUTION_FNS = {
    "alias-substitution": alias_substitution_fn,
    "popularity-substitution": popularity_substitution_fn,
    "corpus-substitution": corpus_substitution_fn,
    "type-swap-substitution": type_swap_substitution_fn,
}


def generate_substitutions(args):
    """Initialize the preprocessed dataset and substitution function, then apply 
    the latter to the former, and save the output as JSONLINES file (one example per line).
    """
    dset_name = os.path.basename(args.inpath).split(".")[0]
    preprocessed_dataset = QADataset.load(dset_name)

    sub_fn = SUBSTITUTION_FNS[args.substitution]
    # Only pass in the arguments from args that are identically named in the function signature
    params = inspect.signature(sub_fn).parameters.values()
    sub_exs = sub_fn(
        preprocessed_dataset,
        args.wikidata,
        **{p.name: vars(args)[p.name] for p in params if p.name in vars(args)},
    )

    # Write final substitution set to args.outpath
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    with open(args.outpath, "w") as outf:
        json.dump({"dataset": f"{dset_name}-{args.substitution}"}, outf)
        outf.write("\n")
        for ex in sub_exs:
            json.dump(ex.json_dump(save_full=args.save_full), outf)
            outf.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Subparses additional arguments for types of `substitution`.",
        dest="substitution",
    )

    parser.add_argument(
        "-i",
        "--inpath",
        type=str,
        required=True,
        help=f"Path to pre-processed dataset, which will be the target of substitution.",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        required=True,
        help="Save path for substitution dataset, once it is generated.",
    )
    parser.add_argument(
        "-w",
        "--wikidata",
        default="wikidata/entity_info.json.gz",
        help="Path to wikidata entity info file generated in Stage 2.",
    )
    parser.add_argument(
        "-s",
        "--save-full",
        type=argparse_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If True, each substitute example, saved at `--out` will record all details of the original example, not just a pointer/id.",
    )
    parser.add_argument(
        "-r",
        "--replace-every",
        type=argparse_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to replace every original answer alias in the context, or just the primary one.",
    )

    # Alias substitution-specific arguments
    alias_sub_parser = subparsers.add_parser("alias-substitution")
    alias_sub_parser.add_argument(
        "-n",
        "--max-aliases",
        type=int,
        default=1,
        help="The maximum number of new examples to generate, capped only by the number of available aliases.",
    )
    alias_sub_parser.add_argument(
        "-c",
        "--category",
        type=str,
        default="NONNUMERIC",
        help="Either `ALL` or one of the answer type categories from your NER model. The `NONNUMERIC` default excludes DATEs and NUMERIC types, which yield poor aliases for substitution.",
    )

    # Popularity substitution-specific arguments
    pop_sub_parser = subparsers.add_parser("popularity-substitution")
    pop_sub_parser.add_argument(
        "-n",
        "--num-bins",
        type=int,
        default=1,
        help="The number of bins which Wikidata entities are split into. Each bin contains an equal number of Wikidata entities group by popularity values. For each original instance, we create a substituted instance for each bin by sampling an entity from the bin.",
    )
    pop_sub_parser.add_argument(
        "-m",
        "--max_ents_per_pop",
        type=int,
        default=10,
        help="The number of entities to keep per popularity value",
    )
    pop_sub_parser.add_argument(
        "-c",
        "--category",
        type=str,
        default="PERSON", # PERSON popularities are more reliable than others categories.
        help="Either `ALL` or one of the answer type categories from your NER model.",
    )

    # Corpus substitution-specific arguments
    corpus_sub_parser = subparsers.add_parser("corpus-substitution")
    corpus_sub_parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=1,
        help="The number of new examples to generate per original example that has an identifiable answer type.",
    )
    corpus_sub_parser.add_argument(
        "-c",
        "--category",
        type=str,
        default="ALL",
        help="Either `ALL` or one of the answer type categories from your NER model.",
    )

    # Type swap substitution-specific arguments
    type_swap_sub_parser = subparsers.add_parser("type-swap-substitution")
    type_swap_sub_parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=1,
        help="The number of new examples to generate per original example that has an identifiable answer type.",
    )
    type_swap_sub_parser.add_argument(
        "-c",
        "--category",
        type=str,
        default="ALL",
        help="Either `ALL` or one of the answer type categories from your NER model.",
    )

    args = parser.parse_args()
    generate_substitutions(args)
