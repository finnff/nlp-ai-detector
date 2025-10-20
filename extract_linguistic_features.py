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

CALCULATE_PERPLEXITY_LOCALLY = False  # Deprecated: perplexity is now pre-computed

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
    """
    Extract perplexity features from pre-computed values in the dataset.

    This function now checks if the dataset already contains perplexity values
    (from the enhanced datasets created by preprocess_perplexity_sources.py)
    and falls back to legacy perplexity files if needed.
    """
    # Check if the main dataset already has perplexity column
    if os.path.exists(DEFAULT_DATASET):
        try:
            if DEFAULT_DATASET.endswith('.csv'):
                df = pd.read_csv(DEFAULT_DATASET)
            else:
                # Handle Arrow format
                from datasets import load_from_disk
                dataset = load_from_disk(DEFAULT_DATASET)
                df = dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()

            if 'perplexity' in df.columns:
                print("✅ Using pre-computed perplexity values from enhanced dataset")
                # Reset index to ensure proper alignment with feature extraction
                df = df.reset_index()
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'orig_index'})
                else:
                    df['orig_index'] = df.index
                return df[['orig_index', 'perplexity']]
            else:
                print("⚠️  Dataset does not contain pre-computed perplexity values")
        except Exception as e:
            print(f"⚠️  Error checking dataset for perplexity: {e}")

    # Legacy fallback - check for separate perplexity files
    print("🔄 Looking for legacy perplexity files...")

    if os.path.exists(perplexity_file):
        print(f"📄 Using legacy perplexity file: {perplexity_file}")
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
            print(f"📄 Using latest legacy perplexity file: {perplexity_file}")
            ppl_df = pd.read_csv(perplexity_file)
            return ppl_df[['orig_index', 'perplexity']]
        else:
            print(f"⚠️  No perplexity values found. Please run preprocess_perplexity_sources.py first")
            print("   or ensure your dataset contains pre-computed perplexity values")
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
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_process', type=int, default=-1)
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for spacy processing')
    args = parser.parse_args()

    # Auto-detect optimal number of processes
    if args.n_process == -1:
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            args.n_process = cpu_count if cpu_count > 12 else 6
            print(f"🖥️  Auto-detected {cpu_count} CPU cores, using {args.n_process} processes")
        except:
            args.n_process = 6
            print(f"🖥️  Could not detect CPU count, using default {args.n_process} processes")

    print(f"🚀 Starting feature extraction...")
    print(f"📁 Dataset: {args.dataset}")
    print(f"💾 Output: {args.output}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"⚡ Processes: {args.n_process}")
    

    print(f"Loading dataset from {args.dataset}")
    if args.dataset.endswith('.csv'):
        df = pd.read_csv(args.dataset)
    else:
        # TODO: Handle arrow format if needed
        raise ValueError(f"Unsupported dataset format: {args.dataset}")

    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    print("\n📊 Extracting linguistic features...")
    feature_rows = []
    valid_indices = []  # Track which original indices we kept

    texts = df['text'].astype(str).tolist()
    labels = df['generated'].tolist()

    print(f"📝 Processing {len(texts)} texts with spaCy (batch mode)...")
    print(f"⚡ Using batch size: {args.batch_size}, processes: {args.n_process}")

    # Use fast batch processing like the original implementation
    try:
        processed_count = 0
        error_count = 0

        # Fast batch processing with nlp.pipe()
        for idx, (row, doc) in enumerate(tqdm(
            zip(df.itertuples(index=False), nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process)),
            total=len(df), desc="🔍 Extracting features"
        )):
            text = row.text
            label = row.generated

            if not text.strip():
                continue

            try:
                features = extract_all_features(text=text, doc=doc)
                features['generated'] = label
                feature_rows.append(features)
                valid_indices.append(idx)
                processed_count += 1

  
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"⚠️  Error processing sample {idx}: {e}")
                elif error_count == 6:
                    print("⚠️  Additional errors suppressed...")
                continue

        print(f"📈 Feature extraction completed: {processed_count} samples processed, {error_count} errors")

    except KeyboardInterrupt:
        print(f"\n⏹️  Processing interrupted at {processed_count}/{len(df)} samples")
        print("🔄 Saving progress so far...")

    except Exception as e:
        print(f"❌ Critical error during feature extraction: {e}")
        print("🚫 Cannot extract features automatically")
        return

    features_df = pd.DataFrame(feature_rows)
    features_df['orig_index'] = valid_indices

    # Add pre-computed perplexity features
    print("🧠 Adding pre-computed perplexity features...")
    perplexity_df = extract_perplexity_features_for_all_samples()

    if not perplexity_df.empty:
        features_df = features_df.merge(
            perplexity_df,
            on='orig_index',
            how='left'
        )
        print(f"✅ Added perplexity features for {perplexity_df.shape[0]} samples")

        # Check perplexity statistics
        if 'perplexity' in features_df.columns:
            non_nan_ppl = features_df['perplexity'].notna().sum()
            total_samples = len(features_df)
            if non_nan_ppl == total_samples:
                print(f"🎯 Perplexity available for all {total_samples} samples")
                ppl_stats = features_df['perplexity'].describe()
                print(f"   Range: {ppl_stats['min']:.2f} - {ppl_stats['max']:.2f} (mean: {ppl_stats['mean']:.2f})")
            else:
                print(f"⚠️  Perplexity available for {non_nan_ppl}/{total_samples} samples ({non_nan_ppl/total_samples*100:.1f}%)")
        else:
            print("⚠️  Perplexity column not found after merging")
    else:
        print("⚠️  No perplexity features available - XGBoost performance may be limited")

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
