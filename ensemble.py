from sklearn.metrics import accuracy_score, classification_report

class Ensemble:
    def __init__(self, classifiers, method='voting'):
        """
        Ensemble class for combining multiple classifiers.

        Args:
            classifiers: Dict of {name: trained_classifier}
            method: 'voting' for majority vote
        """
        self.classifiers = classifiers
        self.method = method

    def fit(self, X, y):
        """
        Fit the ensemble (placeholder for future stacking).
        """
        pass

    def predict(self, X):
        """
        Predict using the ensemble method.
        """
        if self.method == 'voting':
            # Get predictions from each classifier
            preds = {name: clf.predict(X) for name, clf in self.classifiers.items()}
            y_pred = []
            for i in range(len(X)):
                votes = [preds[name][i] for name in preds]
                majority = max(set(votes), key=votes.count)
                y_pred.append(majority)
            return y_pred
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate(self, y_test, y_pred, target_names=['Human Written', 'AI Generated']):
        """
        Evaluate the ensemble predictions.
        """
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        return accuracy, report