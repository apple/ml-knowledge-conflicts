pip install -r requirements.txt

mkdir models
mkdir wikidata

# Download the SpaCy Named Entity Recognizer (NER) and Entity Linker (EL) model
# See https://spacy.io/usage/linguistic-features#named-entities and https://v2.spacy.io/usage/training#entity-linker
wget https://docs-assets.developer.apple.com/ml-research/models/kc-ner/model.gz -O models/kc-ner-model.gz
tar -xvzf models/kc-ner-model.gz -C models/

# Download the Wikidata entity info (output of Optional Stage: Download and Process Wikidata)
wget https://docs-assets.developer.apple.com/ml-research/models/kc-ner/entity_info.json.gz -O wikidata/entity_info.json.gz
