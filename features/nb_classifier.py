import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class NBClassifier:
    def __init__(self, alpha=1.0):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB(alpha=alpha)

    def fit(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        return self

    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_vec)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'])
        return accuracy, report


