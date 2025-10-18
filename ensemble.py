from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class Ensemble:
    def __init__(self, classifiers, method='voting', stacking_estimator='LogisticRegression'):
        """
        Ensemble class for combining multiple classifiers.

        Args:
            classifiers: Dict of {name: trained_classifier}
            method: 'voting' for majority vote, 'stacking' for meta-classifier
            stacking_estimator: Meta-classifier for stacking ('LogisticRegression', 'RandomForest', 'XGBoost')
        """
        self.classifiers = classifiers
        self.method = method
        self.stacking_estimator = stacking_estimator

    def fit(self, X, y):
        """
        Fit the ensemble.
        """
        if self.method == 'stacking':
            estimators = [(name, clf) for name, clf in self.classifiers.items()]
            estimator_map = {
                "LogisticRegression": lambda: LogisticRegression(),
                "RandomForest": lambda: RandomForestClassifier(),
                "XGBoost": lambda: __import__('xgboost').XGBClassifier()
            }
            stacking_estimator = getattr(self, 'stacking_estimator', 'LogisticRegression')
            try:
                final_estimator = estimator_map[stacking_estimator]()
            except KeyError:
                print(f"Warning: Unknown stacking_estimator '{stacking_estimator}', using LogisticRegression")
                final_estimator = LogisticRegression()
            except ImportError:
                print(f"Warning: {stacking_estimator} not available, using LogisticRegression")
                final_estimator = LogisticRegression()
            self.ensemble = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
            self.ensemble.fit(X, y)

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
        elif self.method == 'stacking':
            return self.ensemble.predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate(self, y_test, y_pred, target_names=['Human Written', 'AI Generated']):
        """
        Evaluate the ensemble predictions.
        """
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        return accuracy, report