import os
import tomllib
from datasets import load_from_disk, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from features.nb_classifier import NBClassifier
from features.random_choice import RandomClassifier
from features.nb_no_stopwords import NBClassifierNoStopwords

# Check for combined dataset files
arrow_path = 'data/datasets/combined_dataset'
csv_path = 'data/datasets/combined_dataset.csv'

arrow_exists = os.path.exists(arrow_path)
csv_exists = os.path.exists(csv_path)

if not arrow_exists and not csv_exists:
    print("No combined dataset found (neither Arrow nor CSV)")
    exit(1)

if arrow_exists and csv_exists:
    arrow_mtime = os.path.getmtime(arrow_path)
    csv_mtime = os.path.getmtime(csv_path)
    if arrow_mtime > csv_mtime:
        use_arrow = True
        reason = f"Arrow is newer ({arrow_mtime} > {csv_mtime})"
    else:
        use_arrow = False
        reason = f"CSV is newer ({csv_mtime} > {arrow_mtime})"
elif arrow_exists:
    use_arrow = True
    reason = "Only Arrow version exists"
else:
    use_arrow = False
    reason = "Only CSV version exists"

print(f"Using {'Arrow' if use_arrow else 'CSV'} version: {reason}")

if use_arrow:
    ds = load_from_disk(arrow_path)
else:
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)

print(f"Loaded dataset with {len(ds)} samples")

# Load configuration
with open('configuration.toml', 'rb') as f:
    config = tomllib.load(f)

print(f"Configuration: {config['Title']}")
print(f"Num samples: {config['num_samples']}")
print(f"Random state: {config['random_state']}")
print(f"Test size: {config['test_size']}")

# Limit samples if specified
if config['num_samples'] > 0:
    ds = ds.shuffle(seed=config['random_state']).select(range(min(config['num_samples'], len(ds))))
    print(f"Limited to {len(ds)} samples")

# Extract texts and labels
texts = ds['text']
labels = ds['generated']

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=config['test_size'], random_state=config['random_state'])
print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Define features
features = {
    'nb_classifier': {'class': NBClassifier, 'name': 'NB Classifier'},
    'nb_no_stopwords': {'class': NBClassifierNoStopwords, 'name': 'NB Classifier without Stopwords'},
    'random_choice': {'class': RandomClassifier, 'name': 'Random Classifier'},
}

predictions = {}

for feat, info in features.items():
    if config['features'][feat]['enabled']:
        print(f"\n{info['name']}:")
        params = config['features'][feat].get('params', {})
        clf = info['class'](**params)
        if hasattr(clf, 'fit'):
            clf.fit(X_train, y_train)
            print("Model trained")
        y_pred_feat = clf.predict(X_test)
        predictions[feat] = y_pred_feat
        accuracy_feat, report_feat = clf.evaluate(y_test, y_pred_feat)
        print(f"Accuracy: {accuracy_feat:.4f}")
        print("\nClassification Report:")
        print(report_feat)

# Ensemble
voters = [feat for feat in predictions if config['features'][feat]['allow_voting']]
if len(voters) > 1:
    print("\nEnsemble (Majority Vote):")
    y_pred_ensemble = []
    for i in range(len(y_test)):
        votes = [predictions[feat][i] for feat in voters]
        majority = max(set(votes), key=votes.count)
        y_pred_ensemble.append(majority)

    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    report_ensemble = classification_report(y_test, y_pred_ensemble, target_names=['Human Written', 'AI Generated'])
    print(f"Accuracy: {accuracy_ensemble:.4f}")
    print("\nClassification Report:")
    print(report_ensemble)
