import os
import tomllib
from datasets import load_from_disk, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from features.nb_classifier import NBClassifier
from features.random_choice import RandomClassifier
from features.nb_no_stopwords import NBClassifierNoStopwords
from features.bert_classifier import BERTClassifier
from ensemble import Ensemble
import argparse
import datetime

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
    'bert_classifier': {'name': 'BERT Classifier', 'class': BERTClassifier}
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

    # Limit samples if specified
    if config['num_samples'] > 0:
        ds = ds.shuffle(seed=config['random_state']).select(range(min(config['num_samples'], len(ds))))
        msg = f"Limited to {len(ds)} samples"
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
    trained_clfs = {}

    for feat, info in features.items():
        if config['features'][feat]['enabled']:
            msg = f"\n{info['name']}:"
            print(msg)
            print(msg, file=f)

            params = config['features'][feat].get('params', {})

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
                        accuracy_feat, report_feat = clf.evaluate(all_y_test, all_y_pred)
                        msg = f"Cross-validation completed with {cv_folds} folds"
                        print(msg)
                        print(msg, file=f)
                        # Predict on test set for voting using last clf
                        y_pred_feat = clf.predict(X_test)
                        predictions[feat] = y_pred_feat
                        trained_clfs[feat] = clf
                        # Do not save model in CV mode
                    else:
                        # Train new model
                        clf = info['class'](**params)
                        clf.fit(X_train, y_train)
                        y_pred_feat = clf.predict(X_test)
                        accuracy_feat, report_feat = clf.evaluate(y_test, y_pred_feat)
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
                    clf.fit(X_train, y_train)
                    msg = "Model trained"
                    print(msg)
                    print(msg, file=f)

            if feat != 'bert_classifier' or not use_cv:
                if 'y_pred_feat' not in locals():
                    y_pred_feat = clf.predict(X_test)
                    predictions[feat] = y_pred_feat
                    accuracy_feat, report_feat = clf.evaluate(y_test, y_pred_feat)
                trained_clfs[feat] = clf

            msg = f"Accuracy: {accuracy_feat:.4f}"
            print(msg)
            print(msg, file=f)

            msg = "\nClassification Report:"
            print(msg)
            print(msg, file=f)
            print(report_feat)
            print(report_feat, file=f)

    # Ensemble
    voters = [feat for feat in trained_clfs if config['features'][feat]['allow_voting']]
    if len(voters) > 1:
        ensemble = Ensemble({feat: trained_clfs[feat] for feat in voters}, method='voting')
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        accuracy_ensemble, report_ensemble = ensemble.evaluate(y_test, y_pred_ensemble)

        msg = "\nEnsemble (Majority Vote):"
        print(msg)
        print(msg, file=f)

        msg = f"Accuracy: {accuracy_ensemble:.4f}"
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
