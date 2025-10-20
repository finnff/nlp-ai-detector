import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import xgboost as xgb
import os

class XGBoostClassifier:
    def __init__(self, features_file='data/features/extracted_features.csv', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, test_size=0.2):
        self.features_file = features_file
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        XGBoost works with pre-extracted features passed directly from main.py.
        No need to load features file or create splits - main.py handles this.
        """
        print(f"✅ Training XGBoost on {len(X_train)} feature samples")

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test):
        """
        Make predictions on pre-extracted features from main.py.
        Simple and efficient - no feature extraction needed.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = self.model.predict(X_test)
        return predictions.tolist()

    def get_feature_importances(self):
        """
        Return feature importances as a formatted DataFrame.
        """
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None

        try:
            # Load feature names from the features file
            df = pd.read_csv(self.features_file)
            feature_names = df.drop('generated', axis=1).columns

            importances = self.model.feature_importances_

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            return importance_df
        except Exception as e:
            print(f"⚠️  Could not load feature importances: {e}")
            return None

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate F1 scores directly for full precision
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        report = classification_report(y_test, y_pred, target_names=['Human', 'AI Generated'], digits=4)
        return accuracy, report, f1_macro, f1_weighted