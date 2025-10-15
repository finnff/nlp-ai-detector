import pandas as pd
import numpy as np
import argparse
import tomllib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='configuration.toml', help='Path to config file')
parser.add_argument('--features', default='data/features/extracted_features.csv')
args = parser.parse_args()

with open(args.config, 'rb') as f_config:
    config = tomllib.load(f_config)

test_size = config['test_size']
random_state = config['random_state']

os.makedirs('results', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results/xgboost_results_{timestamp}.txt"
f = open(output_file, 'w')

msg = f"Loading features from {args.features}"
print(msg)
print(msg, file=f)
df = pd.read_csv(args.features)

msg = f"Dataset shape: {df.shape}"
print(msg)
print(msg, file=f)

msg = f"Columns: {df.columns.tolist()}"
print(msg)
print(msg, file=f)

X = df.drop('generated', axis=1)
y = df['generated']

msg = f"\nFeatures shape: {X.shape}"
print(msg)
print(msg, file=f)

msg = "Class distribution:"
print(msg)
print(msg, file=f)
print(y.value_counts())
print(y.value_counts(), file=f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

msg = f"\nTrain set: {X_train.shape[0]} samples"
print(msg)
print(msg, file=f)

msg = f"Test set: {X_test.shape[0]} samples"
print(msg)
print(msg, file=f)

msg = "\nTraining XGBoost"
print(msg)
print(msg, file=f)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=random_state,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
msg = f"\nTest Accuracy: {accuracy:.4f}"
print(msg)
print(msg, file=f)

msg = "\nClassification Report:"
print(msg)
print(msg, file=f)
report = classification_report(y_test, y_pred, target_names=['Human', 'AI Generated'])
print(report)
print(report, file=f)

msg = "\nConfusion Matrix:"
print(msg)
print(msg, file=f)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(conf_matrix, file=f)

#  Feature Importances
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

msg = "\nFeature Importances:"
print(msg)
print(msg, file=f)
print(importance_df)
print(importance_df, file=f)

f.close()
print(f"\nResults saved to {output_file}")

