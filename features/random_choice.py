import random
from sklearn.metrics import accuracy_score, classification_report, f1_score

class RandomClassifier:
    def predict(self, X_test):
        return [random.choice([0, 1]) for _ in X_test]

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate F1 scores directly for full precision
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'], digits=4)
        return accuracy, report, f1_macro, f1_weighted
