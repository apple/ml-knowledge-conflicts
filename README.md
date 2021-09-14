# Entity-Based Knowledge Conflicts in Question Answering

[**Run Instructions**](#run-instructions) | [**Paper**](http://arxiv.org/abs/2109.05052) | [**Citation**](#citation) | [**License**](#license)

This repository provides the **Substitution Framework** described in Section 2 of our paper Entity-Based Knowledge Conflicts in Question Answering.
Given a quesion answering dataset, we derive a new dataset where the context passages have been modified to have new answers to their question.
By training on the original examples and evaluating on the derived examples, we simulate a parametric-contextual knowledge conflict --- useful for understanding how model's employ sources of knowledge to arrive at a decision.

Our dataset derivation follows two steps: (1) identifying named entity answers, and (2) replacing all occurrences of the answer in the context with a substituted entity, effectively changing the answer.
The answer substitutions depend on the chosen [substitution policy](#our-substitution-functions).

## Run Instructions

### 1. Setup

Setup requirements and download SpaCy and WikiData dependencies. 
```
bash setup.sh
```

### 2. (Optional) Download and Process Wikidata

This optional stage reproduces `wikidata/entity_info.json.gz`, downloaded during Setup.

Download the Wikidata dump from October 2020 [here](https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2) and the Wikipedia pageviews from June 2, 2020 [here](https://dumps.wikimedia.org/other/pageview_complete/2021/2021-06/pageviews-20210602-user.bz2).

**NOTE:** We don't use the newest Wikidata dump because Wikidata doesn't keep old dumps so reproducibility is an issue.
If you'd like to use the newest dump, it is available [here](https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2).
Wikipedia pageviews, on the other hand, are kept around and can be found [here](https://dumps.wikimedia.org/other/pageview_complete/).
Be sure to download the `*-user.bz2` file and not the `*-automatic.bz2` or the `*-spider.bz2` files.

To extract out Wikidata information, run the following (takes ~8 hours)
```
python extract_wikidata_info.py --wikidata_dump wikidata-20201026-all.json.bz2 --popularity_dump pageviews-20210602-user.bz2 --output_file entity_info.json.gz
```

The output file of this step is available [here](https://docs-assets.developer.apple.com/ml-research/models/kc-ner/entity_info.json.gz). 

### 3. Load and Preprocess Dataset

```
PYTHONPATH=. python src/load_dataset.py -d MRQANaturalQuestionsTrain -w wikidata/entity_info.json.gz
PYTHONPATH=. python src/load_dataset.py -d MRQANaturalQuestionsDev -w wikidata/entity_info.json.gz
```

### 4. Generate Substitutions
```
PYTHONPATH=. python src/generate_substitutions.py --inpath datasets/normalized/MRQANaturalQuestionsTrain.jsonl --outpath datasets/substitution-sets/MRQANaturalQuestionsTrain<substitution_type>.jsonl <substitution-command> -n 1 ...
PYTHONPATH=. python src/generate_substitutions.py --inpath datasets/normalized/MRQANaturalQuestionsDev.jsonl --outpath datasets/substitution-sets/MRQANaturalQuestionsDev<substitution_type>.jsonl <substitution-command> -n 1 ...
```

See descriptions of the substitution policies (substitution-commands) we provide [here](#our-substitution-functions).
Inspect the argparse and substitution-specific subparsers in `generate_substitutions.py` to see additional arguments.

## Our Substitution Functions

Here we define the the substitution functions we provide.
These functions ingests a QADataset, and modifies the context passage, according to defined rules, such that there is now a new answer to the question, according to the context.
Greater detail is provided in our paper.

* **Alias Substitution** (sub-command: `alias-substitution`) --- Here we replace an answer with one of it's wikidata aliases. 
Since the substituted answer is always semantically equivalent, answer type preservation is naturally maintained.
* **Popularity Substitution** (sub-command: `popularity-substitution`) --- Here we replace answers with a WikiData answer of the same type, with a specified popularity bracket (according to monthly page views).
* **Corpus Substitution** (sub-command: `corpus-substitution`) --- Here we replace answers with other answers of the same type, sampled from the same corpus.
* **Type Swap Substitution** (sub-command: `type-swap-substitution`) --- Here we replace answers with other answers of different type, sampled from the same corpus.

## How to Add Your own Dataset / Substitution Fn / NER Models

### Use your own Dataset

To add your own dataset, create your own subclass of `QADataset` (in `src/classes/qadataset.py`).

1. Overwrite the `read_original_dataset` function, to read your dataset, creating a List of `QAExample` objects.
2. Add your class and the url/filepath to the `DATASETS` variable in `src/load_dataset.py`.

See `MRQANaturalQuetsionsDataset` in `src/classes/qadataset.py` as an example.

### Use your own Substitution Function

We define 5 different substitution functions in `src/generate_substitutions.py`. These are described [here](#our-substitution-functions).
Inspect their docstrings and feel free to add your own, leveraging any of the wikidata, derived answer type, or other info we populate for examples and answers.
Here are the steps to create your own:

1. Add a subparser in `src/generate_substitutions.py` for your new function, with any relevant parameters. See `alias_sub_parser` as an example.
2. Add your own substitution function to `src/substitution_fns.py`, ensuring the signature arguments match those specified in the subparser. See `alias_substitution_fn` as an example.
3. Add a reference to your new function to `SUBSTITUTION_FNS` in `src/generate_substitutions.py`. Ensure the dictionary key matches the subparser name.

### Use your own Named Entity Recognition and/or Entity Linking Model

Our SpaCy NER model is trained and used mainly to categorize answer text into answer types. 
Only substitutions that preserve answer type are likely to be coherent.

The functions which need to be changed are:
1. `run_ner_linking` in `utils.py`, which loads the NER model and populates info for each answer (see function docstring).
2. `Answer._select_answer_type()` in `src/classes/answer.py`, which uses the NER answer type label and wikidata type labels to cateogrize the answer into a type category.

## Citation

Please cite the following if you found this resource or our paper useful.
```
@misc{longpre2021entitybased,
      title={Entity-Based Knowledge Conflicts in Question Answering}, 
      author={Shayne Longpre and Kartik Perisetla and Anthony Chen and Nikhil Ramesh and Chris DuBois and Sameer Singh},
      year={2021},
      eprint={2109.05052},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
The Knowledge Conflicts repository, and entity-based substitution framework are licensed according to the [LICENSE](LICENSE) file.


## Contact Us
To contact us feel free to email the authors in the paper or create an issue in this repository.
