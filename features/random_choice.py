import random
from sklearn.metrics import accuracy_score, classification_report

class RandomClassifier:
    def predict(self, X_test):
        return [random.choice([0, 1]) for _ in X_test]

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'])
        return accuracy, report
