"""Extracts Wikidata entity information from various dumps and outputs to a
single JSON file. The JSON file is broken down into a line for each entity
for easy reading. Each line has the following format:

entity_id: {
    "label": ``str`` Name of entity,
    "aliases": ``str`` Alternative names of entities,
    "entity_types: ``list[str]`` List of entity types,
    "wikipedia_page": ``str`` Wikipedia page of entity,
    "popularity": ``int`` Number of page views for Wikipedia page for one day
}
"""
import argparse
import bz2
import collections
import gzip
import os

import tqdm

from src.utils import BasicTimer

try:
    import ujson as json
except ImportError:
    import json




def extract_label(line):
    """Extracts the English label (canonical name) for an entity"""
    if "en" in line["labels"]:
        return line["labels"]["en"]["value"]
    return None


def extract_aliases(line):
    """Extracts alternative English names for an entity"""
    if "en" in line["aliases"]:
        return [d["value"] for d in line["aliases"]["en"]]
    return []


def extract_entity_types(line):
    """Extracts the entity type for an entity"""
    entity_types = []
    if "P31" in line["claims"]:
        for d in line["claims"]["P31"]:
            if "mainsnak" in d and "datavalue" in d["mainsnak"]:
                entity_types.append(d["mainsnak"]["datavalue"]["value"]["id"])
    return entity_types


def extract_wikipedia_page(line):
    """Extracts the Wikipedia page for an entity"""
    if "sitelinks" in line and "enwiki" in line["sitelinks"]:
        return line["sitelinks"]["enwiki"]["title"].strip().replace(" ", "_")
    return None


def extract_popularities(popularity_dump):
    """Iterate through the Wikipedia popularity dump without decompressing
    it, storing each English Wikipedia page's number of page views.

    Args:
        popularity_dump: ``str`` A path to a .BZ2 file containing Wikipedia
        page views for a day.

    Returns:
        wiki_popularity: ``dict`` Maps from a Wikipedia page to the daily
        page view count.
    """
    wiki_popularity = collections.defaultdict(int)
    with bz2.open(popularity_dump, "rt") as bz_file:
        # Each line corresponds to the number of page views for a Wikipedia page
        for line in tqdm.tqdm(bz_file, desc="Loading Wikipedia popularity values"):
            line = line.strip().split()
            # Skip lines w/o right len or Wikipedia pages that aren't in English
            if len(line) == 6 and line[0] == "en.wikipedia":
                wiki_popularity[line[1]] += int(line[4])
    print(f"Found {len(wiki_popularity)} English Wikipedia pages")
    return wiki_popularity


def extract_entity_information(popularity_dump, wikidata_dump, output_file):
    """For each Wikidata entity in the Wikidata dump, we extract out it's entity
    type, associated Wikipedia page (used for popularity), all aliases
    for the entity, and popularity of the entity's Wikipedia page, then write
    this information into a JSON file. We write each dictionary of entity
    information in it's own line for easy readability.

    Args:
        popularity_dump: ``str``: Path to the Wikipedia popularity dump
        wikidata_dump: ``str`` Path to the Wikidata dump
        output_file: ``str`` Output JSON file
    """
    timer = BasicTimer(f"Extracting Wikidata entities information")
    # Iterate through the Wikipedia popularity dump without decompressing it,
    # storing each English Wikipedia page's number of page views.
    wiki_popularity = extract_popularities(popularity_dump)

    # Iterate through the Wikidata dump without decompressing it
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    writer = gzip.open(output_file, "wb")

    with bz2.open(wikidata_dump, "rt") as bz_file:
        lines_written = 0
        # Each line corresponds to a dictionary about a Wikidata entity
        for line in tqdm.tqdm(bz_file, desc="Processing Wikidata", smoothing=0):
            # The first and last lines of this file are list delimiters, skip these.
            # We also add a hack that checks if the entity has an English Wikipedia
            # page. If not, we skip the line (and thus the JSON loading which is slow)
            # Removing this hack does not change the resulting file.
            line = line.strip()
            if line == "[" or line == "]" or '"enwiki"' not in line:
                continue

            # Remove last character (comma), then decode
            line = json.loads(line[:-1])

            # For each line, extract out relevant Wikidata information
            label = extract_label(line)
            aliases = extract_aliases(line)
            entity_types = extract_entity_types(line)
            wikipedia_page = extract_wikipedia_page(line)
            popularity = wiki_popularity.get(wikipedia_page)

            # Skip if no entity type, label, or popularity value
            if label is None or popularity is None or entity_types == []:
                continue

            entity_dict = {
                "label": label,
                "aliases": aliases,
                "entity_types": entity_types,
                "wikipedia_page": wikipedia_page,
                "popularity": popularity,
            }

            # Write extracted dictionary into a JSON format, one line at a time
            if lines_written > 0:
                writer.write(b",\n")
            writer.write(
                f"{json.dumps(line['id'])}: "
                f"{json.dumps(entity_dict, ensure_ascii=False)}".encode()
            )
            lines_written += 1

    writer.write(b"\n}")
    writer.close()
    timer.finish()


def main():
    """
    For each Wikidata entity in the Wikidata dump, we extract out it's entity
    type, associated Wikipedia page (used for popularity), all aliases
    for the entity, and popularity of the entity's Wikipedia page, then write
    this information into a compressed JSON file. We write each dictionary of entity
    information in it's own line for easy readability.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--wikidata_dump",
        required=True,
        help="Compressed .json.bz2 Wikidata dump for information extraction",
    )
    parser.add_argument(
        "-p",
        "--popularity_dump",
        required=True,
        help="Compressed .bz2 Wikipedia popularity dump",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default="wikidata/entity_info.json.gz",
        help="Output compressed JSON file for writing Wikidata entity information.",
    )
    args = parser.parse_args()

    extract_entity_information(
        popularity_dump=args.popularity_dump,
        wikidata_dump=args.wikidata_dump,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
