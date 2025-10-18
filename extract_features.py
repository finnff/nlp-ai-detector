import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from spacy.parts_of_speech import IDS
from spacy.attrs import POS as POS_ATTR
import re
import spacy
from math import sqrt
from collections import Counter

CALCULATE_PERPLEXITY_LOCALLY = True

DEFAULT_DATASET = 'data/datasets/combined_dataset.csv'
DEFAULT_OUTPUT = 'data/features/extracted_features.csv'
DEFAULT_PERPLEXITY_OUTPUT = 'data/features/perplexity_results.csv'

# load/compute here for efficiency 
nlp = spacy.load("en_core_web_sm", disable=["ner"])
COARSE_POS_LABELS = [nlp.vocab.strings[pos_id] for pos_id in IDS.values()]
COARSE_POS_LABELS = [p for p in COARSE_POS_LABELS if p and p not in ("SPACE", "EOL")]
POS_INDEX = {label: i for i, label in enumerate(COARSE_POS_LABELS)}


def extract_lexical_features(text, doc=None):
    if(doc is None):
        doc = nlp(text)
    
    features = {}
    
    # word length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words_in_text = [sentence.split() for sentence in sentences]
    word_lengths = [len(word) for word in words_in_text]
    
    features['average_word_length'] = sum(word_lengths) / len(word_lengths)
    features['word_length_variance'] = np.var(word_lengths)
    
    # Root Token Type Ratio (normalizing for text length)
    lemmas = ([token.lemma_ for token in doc])
    unique_lemmas = set(lemmas)
    unique_words = len(unique_lemmas)
    features['rttr_root_type_token_ratio'] = unique_words / sqrt(len(doc.text.split()))
    
    # Hapax Legomena Ratiom which measures the proportion of words that appear only once 
    lemmas_count = Counter(([token.lemma_ for token in doc]))
    words_appearing_once = [word for word, count in lemmas_count.items() if count == 1]
    features['hapax_legomena_ratio'] = len(words_appearing_once) / len(doc.text.split())
    
    return features


def extract_syntactic_features(text, doc=None):
    if doc is None:
        doc = nlp(text)
    
    features = {}
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    features['average_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
    features['sentence_length_variance'] = np.var(sentence_lengths)
    
    
    # parse tree depth, syntactic complexity
    sentence_depths = []

    for sent in doc.sents:
        sent_depth = max(len(list(token.ancestors)) for token in sent)
        sentence_depths.append(sent_depth)

    features['average_parse_depth'] = np.mean(sentence_depths)
    features['maximum_parse_depth'] = max(sentence_depths)
    features['parse_depth_variance'] = np.var(sentence_depths)
    
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


def extract_perplexity_features_for_all_samples(perplexity_file=DEFAULT_PERPLEXITY_OUTPUT):
    if CALCULATE_PERPLEXITY_LOCALLY:
        print("Calculating perplexity locally.")

        from perplexity_calc import calculate_perplexity
        output_path = calculate_perplexity(
            dataset_path=DEFAULT_DATASET,
            output_path=DEFAULT_PERPLEXITY_OUTPUT
        )
        perplexity_file = output_path

    if os.path.exists(perplexity_file):
        ppl_df = pd.read_csv(perplexity_file)
        return ppl_df[['orig_index', 'perplexity']]
    else:
        # Look for latest timestamped file
        import glob
        pattern = 'data/features/perplexity_results_*.csv'
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            perplexity_file = files[0]
            print(f"Using latest perplexity file: {perplexity_file}")
            ppl_df = pd.read_csv(perplexity_file)
            return ppl_df[['orig_index', 'perplexity']]
        else:
            print(f"Warning: No perplexity file found at {perplexity_file} or timestamped versions")
            return pd.DataFrame(columns=['orig_index', 'perplexity'])
    

# TODO: discuss whether to include this (my vote: skip.)
def extract_burstiness_features(text):
    features = {}
    return features


def extract_all_features(text=None, doc=None):
    features = {}
    
    features.update(extract_lexical_features(text, doc=doc))
    features.update(extract_syntactic_features(text, doc=doc))
    features.update(extract_punctuation_features(text))
    features.update(extract_pos_features(text=text, doc=doc))
    features.update(extract_burstiness_features(text))
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--n_process', type=int, default=-1)
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
    valid_indices = []  # Track which original indices we kept

    texts = df['text'].astype(str).tolist()
    labels = df['generated'].tolist()

    for idx, (row, doc) in enumerate(tqdm(
        zip(df.itertuples(index=False), nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process)),
        total=len(df), desc="Processing"
    )):
        text = row.text
        label = row.generated

        if not text.strip():
            continue

        features = extract_all_features(text=text, doc=doc)
        features['generated'] = label
        feature_rows.append(features)
        valid_indices.append(idx)

    features_df = pd.DataFrame(feature_rows)
    features_df['orig_index'] = valid_indices
    
    # add perplexity (calculated in "batch")
    perplexity_df = extract_perplexity_features_for_all_samples()
    
    if not perplexity_df.empty:
        features_df = features_df.merge(
            perplexity_df, 
            on='orig_index', 
            how='left'
        )
        print(f"Added perplexity features for {perplexity_df.shape[0]} samples")
    else:
        print("No perplexity features to add")
    
    # Drop orig_index as it was just for merging
    features_df = features_df.drop(columns=['orig_index'])
    

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
