import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

class XGBoostClassifier:
    def __init__(self, features_file='data/features/extracted_features_processed.csv', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, test_size=0.2):
        self.features_file = features_file
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.test_size = test_size
        self.model = None

    def fit(self, X_train, y_train):
        # Load features and split
        df = pd.read_csv(self.features_file)
        X = df.drop('generated', axis=1)
        y = df['generated']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(self.X_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=['Human', 'AI Generated'])
        return accuracy, report