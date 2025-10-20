import pandas as pd
import argparse
import tomllib
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configuration.toml', 
                        help='Path to config file')
    parser.add_argument('--features', default='data/features/extracted_features.csv',
                        help='Path to extracted features CSV file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'rb') as f_config:
        config = tomllib.load(f_config)
    return config


def load_and_split_data(features_path, test_size, random_state):
    df = pd.read_csv(features_path)
    X = df.drop('generated', axis=1)
    y = df['generated']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns


def load_and_split_data_with_logging(features_path, test_size, random_state, file_handle=None):
    dual_print(f"Loading features from {features_path}", file_handle)
    
    df = pd.read_csv(features_path)
    dual_print(f"Dataset shape: {df.shape}", file_handle)
    dual_print(f"Columns: {df.columns.tolist()}", file_handle)
    
    X = df.drop('generated', axis=1)
    y = df['generated']
    
    dual_print(f"\nFeatures shape: {X.shape}", file_handle)
    dual_print("Class distribution:", file_handle)
    dual_print(y.value_counts(), file_handle)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    dual_print(f"\nTrain set: {X_train.shape[0]} samples", file_handle)
    dual_print(f"Test set: {X_test.shape[0]} samples", file_handle)
    
    return X_train, X_test, y_train, y_test, X.columns


def setup_results_file(model_name):
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/{model_name}_results_{timestamp}.txt"
    f = open(output_file, 'w')
    return f, output_file


def dual_print(message, file_handle=None):
    print(message)
    if file_handle is not None:
        print(message, file=file_handle)


def evaluate_model(model, X_test, y_test, file_handle=None):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    dual_print(f"\nTest Accuracy: {accuracy:.4f}", file_handle)
    dual_print("\nClassification Report:", file_handle)
    report = classification_report(y_test, y_pred, target_names=['Human', 'AI Generated'])
    dual_print(report, file_handle)
    
    dual_print("\nConfusion Matrix:", file_handle)
    conf_matrix = confusion_matrix(y_test, y_pred)
    dual_print(conf_matrix, file_handle)
    
    return y_pred, accuracy


def print_feature_importance(feature_names, importances, title, file_handle=None):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    dual_print(f"\n{title}:", file_handle)
    dual_print(importance_df, file_handle)
    
    return importance_df

