import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from spacy.parts_of_speech import IDS
from spacy.attrs import POS as POS_ATTR
import re
import spacy


DEFAULT_DATASET = 'data/datasets/combined_dataset.csv'
DEFAULT_OUTPUT = 'data/features/extracted_features.csv'

# load/compute here for efficiency 
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
COARSE_POS_LABELS = [nlp.vocab.strings[pos_id] for pos_id in IDS.values()]
COARSE_POS_LABELS = [p for p in COARSE_POS_LABELS if p and p not in ("SPACE", "EOL")]
POS_INDEX = {label: i for i, label in enumerate(COARSE_POS_LABELS)}


def extract_lexical_features(text):
    # - Type-token ratio
    # - Unique word ratio
    # - Average word length
    # - Vocabulary richness
    features = {}
    return features


def extract_syntactic_features(text):
    # - Average sentence length
    # - Sentence length variance
    # - Parse tree depth
    # - Clause complexity
    
    features = {}
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    features['average_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
    features['sentence_length_variance'] = np.var(sentence_lengths)
    
    return features

def extract_punctuation_features(text):
    features = {}
    word_count = len(text.split())
    
    punctuation_features = {'comma': ',', 'period': '.', 'exclamation_mark': '!', 'question_mark': '?'}

    for feature, punctuation in punctuation_features.items():
        features[f'{feature}_ratio'] = text.count(punctuation) / word_count
  
    features['punctuation_per_word'] = sum(text.count(punctuation) for punctuation in punctuation_features.values()) / word_count

    return features

def extract_pos_features(text=None, doc=None):
    if doc is None:
        doc = nlp(text)

    pos_counts = doc.count_by(POS_ATTR)
    counts_arr = np.zeros(len(COARSE_POS_LABELS), dtype=np.int32)
    
    for pos_id, c in pos_counts.items():
        label = nlp.vocab.strings[pos_id] 
        idx = POS_INDEX.get(label)
        if idx is not None:
            counts_arr[idx] = c
    total_tokens = int(counts_arr.sum()) or 1

    features = {}
    for label, idx in POS_INDEX.items():
        features[f"pos_ratio_{label.lower()}"] = float(counts_arr[idx]) / total_tokens

    noun_count = int(counts_arr[POS_INDEX.get("NOUN", 0)]) + int(counts_arr[POS_INDEX.get("PROPN", 0)])
    verb_count = int(counts_arr[POS_INDEX.get("VERB", 0)]) + int(counts_arr[POS_INDEX.get("AUX", 0)])
    features["noun_to_verb_ratio"] = (noun_count / verb_count) if verb_count else 0.0
    return features


def extract_perplexity_features(text):
    features = {}
    return features


def extract_burstiness_features(text):
    features = {}
    return features


def extract_all_features(text=None, doc=None):
    features = {}
    
    features.update(extract_lexical_features(text))
    features.update(extract_syntactic_features(text))
    features.update(extract_punctuation_features(text))
    features.update(extract_pos_features(text=text, doc=doc))
    features.update(extract_perplexity_features(text))
    features.update(extract_burstiness_features(text))
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--n_process', type=int, default=4)
    args = parser.parse_args()
    

    print(f"Loading dataset from {args.dataset}")
    if args.dataset.endswith('.csv'):
        df = pd.read_csv(args.dataset)
    else:
        # TODO: Handle arrow format if needed
        raise ValueError(f"Unsupported dataset format: {args.dataset}")

    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    print("\nExtracting features...")
    feature_rows = []

    texts = df['text'].astype(str).tolist()
    labels = df['generated'].tolist()

    for row, doc in tqdm(
        zip(df.itertuples(index=False), nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process)),
        total=len(df), desc="Processing"
    ):
        text = row.text
        label = row.generated

        if not text.strip():
            continue

        features = extract_all_features(text=text, doc=doc)
        features['generated'] = label
        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    print(f"\nSaving features to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_csv(args.output, index=False)

    print(f"\nFeature extraction complete")
    print(f"Shape: {features_df.shape}")
    print(f"Features extracted: {[col for col in features_df.columns]}")
    print(f"\nFeature stats:")
    print(features_df.describe())


if __name__ == "__main__":
    main()