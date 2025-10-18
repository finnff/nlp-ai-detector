import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report

class RandomClassifier(BaseEstimator, ClassifierMixin):
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X_test):
        return [random.choice([0, 1]) for _ in X_test]

    def predict_proba(self, X_test):
        return np.array([[0.5, 0.5] for _ in X_test])

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'])
        return accuracy, report
