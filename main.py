import os
import tomllib
from datasets import load_from_disk, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, KFold
from features.nb_classifier import NBClassifier
from features.random_choice import RandomClassifier
from features.nb_no_stopwords import NBClassifierNoStopwords
from features.bert_classifier import BERTClassifier
from features.deberta_classifier import DeBERTaClassifier
from features.xgboost_classifier import XGBoostClassifier
import argparse
import datetime

def check_perplexity_status(dataset_path):
    """Check if dataset contains pre-computed perplexity values."""
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            # Handle Arrow format
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            df = dataset['train'].to_pandas() if 'train' in dataset else dataset.to_pandas()

        has_perplexity = 'perplexity' in df.columns
        if has_perplexity:
            ppl_stats = df['perplexity'].describe()
            return True, ppl_stats
        else:
            return False, None
    except Exception as e:
        print(f"⚠️  Could not check perplexity status: {e}")
        return False, None

def main(config_path='configuration.toml'):
    # Load configuration
    with open(config_path, 'rb') as f_config:
        config = tomllib.load(f_config)
    
    # Create output directory and file
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/results_{timestamp}.txt"
    f = open(output_file, 'w')
    
    # Define features dictionary
    ## Define features dictionary
    features = {
    'nb_classifier': {'name': 'Naive Bayes', 'class': NBClassifier},
    'random_choice': {'name': 'Random Classifier', 'class': RandomClassifier},
    'nb_no_stopwords': {'name': 'NB No Stopwords', 'class': NBClassifierNoStopwords},
    'bert_classifier': {'name': 'BERT Classifier', 'class': BERTClassifier},
    'deberta_classifier': {'name': 'DeBERTa Classifier', 'class': DeBERTaClassifier},
    'xgboost_classifier': {'name': 'XGBoost Classifier', 'class': XGBoostClassifier}
    }
    
    # Check for combined dataset files
    arrow_path = 'data/datasets/combined_dataset'
    csv_path = 'data/datasets/combined_dataset.csv'

    arrow_exists = os.path.exists(arrow_path)
    csv_exists = os.path.exists(csv_path)

    if not arrow_exists and not csv_exists:
        msg = "No combined dataset found (neither Arrow nor CSV)"
        print(msg)
        print(msg, file=f)
        f.close()
        exit(1)

    current_time = datetime.datetime.now().timestamp()
    if arrow_exists:
        arrow_mtime = os.path.getmtime(arrow_path)
        arrow_age = (current_time - arrow_mtime) / 60
    if csv_exists:
        csv_mtime = os.path.getmtime(csv_path)
        csv_age = (current_time - csv_mtime) / 60

    if arrow_exists and csv_exists:
        if arrow_mtime > csv_mtime:
            use_arrow = True
            reason = f"Arrow is Newer ({arrow_age:.0f} mins old vs {csv_age:.0f} mins old)"
        else:
            use_arrow = False
            reason = f"CSV is Newer ({csv_age:.0f} mins old vs {arrow_age:.0f} mins old)"
    elif arrow_exists:
        use_arrow = True
        reason = f"Only Arrow version exists ({arrow_age:.0f} mins old)"
    else:
        use_arrow = False
        reason = f"Only CSV version exists ({csv_age:.0f} mins old)"

    msg = f"Using {'Arrow' if use_arrow else 'CSV'} version: {reason}"
    print(msg)
    print(msg, file=f)

    if use_arrow:
        ds = load_from_disk(arrow_path)
        if not hasattr(ds, 'select'):
            ds = ds['train']
    else:
        df = pd.read_csv(csv_path)
        ds = Dataset.from_pandas(df)

    msg = f"Loaded dataset with {len(ds)} samples"
    print(msg)
    print(msg, file=f)

    # Check perplexity status
    dataset_path = csv_path if not use_arrow else arrow_path
    has_perplexity, ppl_stats = check_perplexity_status(dataset_path)

    if has_perplexity:
        msg = f"🧠 Dataset contains pre-computed perplexity values!"
        print(msg)
        print(msg, file=f)
        msg = f"   Perplexity range: {ppl_stats['min']:.2f} - {ppl_stats['max']:.2f} (mean: {ppl_stats['mean']:.2f})"
        print(msg)
        print(msg, file=f)
    else:
        msg = f"⚠️  Dataset does not contain pre-computed perplexity values"
        print(msg)
        print(msg, file=f)
        msg = f"   Perplexity will be calculated during feature extraction (slower)"
        print(msg)
        print(msg, file=f)

    # Extract features if enabled and not exist
    if config.get('extract_features', {}).get('enabled', False):
        features_file = 'data/features/extracted_features.csv'

        # Check if we need to extract features
        need_extraction = False

        if not os.path.exists(features_file):
            need_extraction = True
            msg = "📊 Extracting linguistic features (this may take a few minutes)..."
            print(msg)
            print(msg, file=f)
        else:
            # Check if features match dataset size
            try:
                features_df = pd.read_csv(features_file)
                if len(features_df) != len(ds):
                    msg = f"⚠️  Features file length mismatch (dataset: {len(ds)}, features: {len(features_df)})"
                    print(msg)
                    print(msg, file=f)
                    msg = "Regenerating features to match dataset..."
                    print(msg)
                    print(msg, file=f)
                    need_extraction = True
                    # Remove old features file to ensure clean regeneration
                    os.remove(features_file)
                    print(f"🗑️  Removed old features file: {features_file}")
                else:
                    msg = f"✅ Found existing features file ({len(features_df)} samples)"
                    print(msg)
                    print(msg, file=f)

                    # Also check if perplexity is in the features
                    if 'perplexity' not in features_df.columns and has_perplexity:
                        msg = f"⚠️  Dataset has perplexity but features don't - regenerating..."
                        print(msg)
                        print(msg, file=f)
                        need_extraction = True
                        os.remove(features_file)
                        print(f"🗑️  Removed features file missing perplexity: {features_file}")
            except Exception as e:
                msg = f"⚠️  Error reading features file: {e}"
                print(msg)
                print(msg, file=f)
                need_extraction = True

        if need_extraction:
            import extract_linguistic_features as extract_features
            extract_features.main()
            msg = "✅ Feature extraction completed!"
            print(msg)
            print(msg, file=f)

  
        # Final feature summary
        if os.path.exists(features_file):
            try:
                features_df = pd.read_csv(features_file)
                feature_count = len(features_df.columns) - 1  # Exclude 'generated' column
                has_perplexity_feature = 'perplexity' in features_df.columns

                msg = f"📈 Ready features: {feature_count} linguistic features"
                print(msg)
                print(msg, file=f)

                if has_perplexity_feature:
                    msg = f"🧠 Includes pre-computed perplexity values"
                    print(msg)
                    print(msg, file=f)
                else:
                    msg = f"⚠️  Missing perplexity values - XGBoost performance may be limited"
                    print(msg)
                    print(msg, file=f)
            except Exception as e:
                msg = f"⚠️  Could not summarize features: {e}"
                print(msg)
                print(msg, file=f)

    # Feature preprocessing already handled above

    # Limit samples if specified
    if config['num_samples'] > 0:
        ds = ds.shuffle(seed=config['random_state']).select(range(min(config['num_samples'], len(ds))))
        msg = f"Limited to {len(ds)} samples"
        print(msg)
        print(msg, file=f)
    else:
        # When num_samples = 0, still shuffle to match XGBoost features processing
        ds = ds.shuffle(seed=config['random_state'])
        msg = f"Using all {len(ds)} samples (shuffled)"
        print(msg)
        print(msg, file=f)

    # Extract texts and labels
    texts = ds['text']
    labels = ds['generated']

    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    msg = f"Train set: {len(X_train)} samples"
    print(msg)
    print(msg, file=f)

    msg = f"Test set: {len(X_test)} samples"
    print(msg)
    print(msg, file=f)

    # Create XGBoost feature splits using simple OLD approach logic
    features_file = 'data/features/extracted_features.csv'
    if os.path.exists(features_file):
        df_features = pd.read_csv(features_file)

        print("🔧 Creating XGBoost feature splits (HuggingFace-based fix)...")

        # HUGGINGFACE FIX: Apply identical transformations to features as main pipeline
        X = df_features.drop('generated', axis=1)
        y = df_features['generated']

        # Step 1: Apply same HuggingFace-style shuffle using their exact algorithm
        # Convert to dataset to use HuggingFace's shuffle, then back to pandas
        features_ds = Dataset.from_pandas(df_features)  # Use the full features DataFrame
        features_ds = features_ds.shuffle(seed=config['random_state'])

        # Step 2: Apply same HuggingFace-style limit
        if config['num_samples'] > 0:
            features_ds = features_ds.select(range(min(config['num_samples'], len(features_ds))))
            msg = f"🔧 Limited XGBoost features to {len(features_ds)} samples (using HuggingFace shuffle)"
            print(msg)
            print(msg, file=f)
        else:
            # When num_samples = 0, use all samples but still confirm HuggingFace shuffle was applied
            msg = f"🔧 Using all {len(features_ds)} XGBoost features (with HuggingFace shuffle)"
            print(msg)
            print(msg, file=f)

        # Convert back to pandas
        features_df = features_ds.to_pandas()
        X = features_df.drop('generated', axis=1)
        y = features_df['generated']

        # Step 3: Apply same train/test split as dataset
        X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state']
        )

        msg = f"🔧 XGBoost feature splits created: {len(X_train_feat)} train, {len(X_test_feat)} test samples"
        print(msg)
        print(msg, file=f)
    else:
        msg = f"⚠️  Features file not found: {features_file}"
        print(msg)
        print(msg, file=f)
        X_train_feat, X_test_feat, y_train_feat, y_test_feat = None, None, None, None

    msg = "\nEnabled Features:"
    print(msg)
    print(msg, file=f)

    for feat, info in features.items():
        if config['features'][feat]['enabled']:
            params = config['features'][feat].get('params', {})
            allow_voting = config['features'][feat]['allow_voting']
            extra = ""
            if feat == 'bert_classifier':
                use_cv = config['features'][feat].get('use_cv', False)
                extra = f", use_cv={use_cv}"
            msg = f"- {info['name']}: params={params}, allow_voting={allow_voting}{extra}"
            print(msg)
            print(msg, file=f)

    predictions = {}

    ### absolutely disguisting code but works for now ###

    for feat, info in features.items():
        if config['features'][feat]['enabled']:
            msg = f"\n{info['name']}:"
            print(msg)
            print(msg, file=f)

            params = config['features'][feat].get('params', {})
            if feat == 'xgboost_classifier':
                params['test_size'] = config['test_size']
                params['random_state'] = config['random_state']

            use_cv = False
            cv_folds = 5

            # Special handling for BERT classifier with pretrained model option
            if feat == 'bert_classifier':
                use_pretrained = config['features'][feat].get('use_pretrained', False)
                model_path = config['features'][feat].get('model_path', 'models/bert_classifier.pt')
                save_after_training = config['features'][feat].get('save_after_training', False)
                use_cv = config['features'][feat].get('use_cv', False)
                cv_folds = config['features'][feat].get('cv_folds', 5)

                if use_pretrained:
                    # Load pretrained model
                    try:
                        clf = info['class'].from_pretrained(
                            model_path,
                            model_name=params.get('model_name', 'google-bert/bert-base-uncased'),
                            lr=params.get('lr', 1e-3)
                        )
                        msg = f"Loaded pretrained model from {model_path}"
                        print(msg)
                        print(msg, file=f)
                    except FileNotFoundError:
                        msg = f"Error: Pretrained model not found at {model_path}. Please train a model first or set use_pretrained = false."
                        print(msg)
                        print(msg, file=f)
                        continue
                else:
                    if use_cv:
                        # Cross-validation
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=config['random_state'])
                        all_y_test = []
                        all_y_pred = []
                        clf = None
                        for fold, (train_idx, test_idx) in enumerate(kf.split(texts)):
                            X_train_cv = [texts[i] for i in train_idx]
                            X_test_cv = [texts[i] for i in test_idx]
                            y_train_cv = [labels[i] for i in train_idx]
                            y_test_cv = [labels[i] for i in test_idx]
                            clf = info['class'](**params)
                            clf.fit(X_train_cv, y_train_cv)
                            y_pred_cv = clf.predict(X_test_cv)
                            all_y_test.extend(y_test_cv)
                            all_y_pred.extend(y_pred_cv)
                            msg = f"Fold {fold+1}/{cv_folds} completed"
                            print(msg)
                            print(msg, file=f)
                        # Evaluate on all CV predictions
                        accuracy_feat, report_feat, f1_macro_feat, f1_weighted_feat = clf.evaluate(all_y_test, all_y_pred)
                        msg = f"Cross-validation completed with {cv_folds} folds"
                        print(msg)
                        print(msg, file=f)
                        msg = f"F1 Score (Macro): {f1_macro_feat:.4f}"
                        print(msg)
                        print(msg, file=f)
                        msg = f"F1 Score (Weighted): {f1_weighted_feat:.4f}"
                        print(msg)
                        print(msg, file=f)
            # Special handling for DeBERTa classifier with pretrained model option
            elif feat == 'deberta_classifier':
                use_pretrained = config['features'][feat].get('use_pretrained', False)
                model_path = config['features'][feat].get('model_path', 'models/deberta_classifier.pt')
                save_after_training = config['features'][feat].get('save_after_training', False)

                if use_pretrained:
                    # Load pretrained model
                    try:
                        clf = info['class'].from_pretrained(
                            model_path,
                            model_name=params.get('model_name', 'microsoft/deberta-v3-base'),
                            lr=params.get('lr', 1e-3)
                        )
                        msg = f"Loaded pretrained model from {model_path}"
                        print(msg)
                        print(msg, file=f)
                    except FileNotFoundError:
                        msg = f"Error: Pretrained model not found at {model_path}. Please train a model first or set use_pretrained = false."
                        print(msg)
                        print(msg, file=f)
                        continue
                else:
                    # Train new model
                    clf = info['class'](**params)
                    clf.fit(X_train, y_train)
                    msg = "Model trained"
                    print(msg)
                    print(msg, file=f)

                    # Save model if requested
                    if save_after_training:
                        clf.save_model(model_path)
                        msg = f"Model saved to {model_path}"
                        print(msg)
                        print(msg, file=f)
            else:
                # Regular handling for other classifiers
                clf = info['class'](**params)

                if hasattr(clf, 'fit'):
                    if feat == 'xgboost_classifier':
                        # XGBoost: Pass features directly instead of text
                        if X_train_feat is not None and X_test_feat is not None:
                            clf.fit(X_train_feat, y_train_feat)
                            msg = "Model trained (using pre-extracted features)"
                            print(msg)
                            print(msg, file=f)
                        else:
                            msg = "⚠️  No features available for XGBoost, skipping..."
                            print(msg)
                            print(msg, file=f)
                            continue
                    else:
                        # Other classifiers: Use text as usual
                        clf.fit(X_train, y_train)
                        msg = "Model trained"
                        print(msg)
                        print(msg, file=f)

            if feat != 'bert_classifier' or not use_cv:
                # Each classifier should generate its own predictions
                if feat == 'xgboost_classifier':
                    # XGBoost: Use features for prediction
                    if X_test_feat is not None:
                        y_pred_feat = clf.predict(X_test_feat)
                        predictions[feat] = y_pred_feat
                    else:
                        msg = "⚠️  No test features available for XGBoost"
                        print(msg)
                        print(msg, file=f)
                        continue
                else:
                    # Other classifiers: Use text as usual
                    y_pred_feat = clf.predict(X_test)
                    predictions[feat] = y_pred_feat
                accuracy_feat, report_feat, f1_macro_feat, f1_weighted_feat = clf.evaluate(y_test, y_pred_feat)

            msg = f"Accuracy: {accuracy_feat:.4f}"
            print(msg)
            print(msg, file=f)

            msg = f"F1 Score (Macro): {f1_macro_feat:.4f}"
            print(msg)
            print(msg, file=f)

            msg = f"F1 Score (Weighted): {f1_weighted_feat:.4f}"
            print(msg)
            print(msg, file=f)

            msg = "\nClassification Report:"
            print(msg)
            print(msg, file=f)
            print(report_feat)
            print(report_feat, file=f)

            # Special handling for XGBoost - show feature importances
            if feat == 'xgboost_classifier':
                importance_df = clf.get_feature_importances()
                if importance_df is not None:
                    msg = "\nFeature Importances:"
                    print(msg)
                    print(msg, file=f)
                    print(importance_df)
                    print(importance_df, file=f)

    # Ensemble
    voters = [feat for feat in predictions if config['features'][feat]['allow_voting']]
    if len(voters) > 1:
        msg = "\nEnsemble (Majority Vote):"
        print(msg)
        print(msg, file=f)
        
        y_pred_ensemble = []
        for i in range(len(y_test)):
            votes = [predictions[feat][i] for feat in voters]
            majority = max(set(votes), key=votes.count)
            y_pred_ensemble.append(majority)

        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

        # Calculate F1 scores directly for full precision
        f1_macro_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')
        f1_weighted_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
        f1_per_class_ensemble = f1_score(y_test, y_pred_ensemble, average=None)

        report_ensemble = classification_report(
            y_test, y_pred_ensemble,
            target_names=['Human Written', 'AI Generated'],
            digits=4
        )
        
        msg = f"Accuracy: {accuracy_ensemble:.4f}"
        print(msg)
        print(msg, file=f)

        msg = f"F1 Score (Macro): {f1_macro_ensemble:.4f}"
        print(msg)
        print(msg, file=f)

        msg = f"F1 Score (Weighted): {f1_weighted_ensemble:.4f}"
        print(msg)
        print(msg, file=f)

        msg = f"F1 Score (Human): {f1_per_class_ensemble[0]:.4f}"
        print(msg)
        print(msg, file=f)

        msg = f"F1 Score (AI Generated): {f1_per_class_ensemble[1]:.4f}"
        print(msg)
        print(msg, file=f)

        msg = "\nClassification Report:"
        print(msg)
        print(msg, file=f)
        print(report_ensemble)
        print(report_ensemble, file=f)

    f.close()
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configuration.toml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
