import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class NBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB(alpha=alpha)

    def get_params(self, deep=True):
        return {'alpha': self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.classifier = MultinomialNB(alpha=self.alpha)
        return self

    def fit(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_vec)

    def predict_proba(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict_proba(X_test_vec)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'])
        return accuracy, report


