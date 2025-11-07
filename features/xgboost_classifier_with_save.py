import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class XGBoostClassifierWithSave:
    def __init__(self, features_file='data/features/extracted_features.csv', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, test_size=0.2):
        self.features_file = features_file
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None
        self.use_scaler = False

    def load_features(self):
        """Load features from CSV file"""
        try:
            df = pd.read_csv(self.features_file)
            X = df.drop('generated', axis=1)
            y = df['generated']
            self.feature_names = X.columns.tolist()
            return X, y
        except Exception as e:
            raise ValueError(f"Could not load features from {self.features_file}: {e}")

    def fit(self, X_train=None, y_train=None, scale_features=False):
        """
        Train XGBoost on features. If X_train and y_train are None, loads from features_file.
        """
        if X_train is None or y_train is None:
            print("Loading features from file...")
            X, y = self.load_features()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )

        print(f"✅ Training XGBoost on {len(X_train)} feature samples")

        # Scale features if requested
        if scale_features:
            self.use_scaler = True
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            print("✅ Features scaled using StandardScaler")

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Store feature names if not already set
        if self.feature_names is None and hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        elif self.feature_names is None and isinstance(X_train, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    def predict(self, X_test):
        """Make predictions on features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Apply scaling if used during training
        if self.use_scaler and self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        predictions = self.model.predict(X_test)
        return predictions.tolist()

    def predict_proba(self, X_test):
        """Make probability predictions on features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Apply scaling if used during training
        if self.use_scaler and self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        probabilities = self.model.predict_proba(X_test)
        return probabilities

    def get_feature_importances(self):
        """Return feature importances as a formatted DataFrame"""
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None

        if self.feature_names is None:
            return None

        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def evaluate(self, X_test, y_test, y_scores=None):
        """
        Enhanced evaluation method with AUROC support.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Apply scaling if used during training
        if self.use_scaler and self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate F1 scores
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        report = classification_report(y_test, y_pred, target_names=['Human', 'AI Generated'], digits=4)

        # Calculate AUROC
        auroc = None
        if y_scores is None:
            y_scores = self.predict_proba(X_test)[:, 1]
        try:
            auroc = roc_auc_score(y_test, y_scores)
        except Exception as e:
            print(f"⚠️  Could not calculate AUROC: {e}")

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auroc': auroc,
            'predictions': y_pred,
            'probabilities': self.predict_proba(X_test)
        }

    def save_model(self, save_path=None, save_format='pt'):
        """
        Save XGBoost model in PyTorch-compatible format (.pt) or traditional format (.pkl)

        Args:
            save_path: Path to save the model (without extension)
            save_format: 'pt' for PyTorch format or 'pkl' for pickle format
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"models/xgboost_classifier_{timestamp}"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_format == 'pt':
            # Save in PyTorch-compatible format
            model_data = {
                'xgb_model': self.model,
                'feature_names': self.feature_names,
                'hyperparameters': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'random_state': self.random_state
                },
                'model_info': {
                    'model_type': 'XGBoost',
                    'is_fitted': self.is_fitted,
                    'use_scaler': self.use_scaler,
                    'num_features': len(self.feature_names) if self.feature_names else None
                },
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'features_file': self.features_file
                }
            }

            # Save scaler if used
            if self.use_scaler and self.scaler is not None:
                model_data['scaler'] = self.scaler

            # Save as .pt file
            torch.save(model_data, f"{save_path}.pt")
            print(f"✅ Model saved as {save_path}.pt (PyTorch format)")

        elif save_format == 'pkl':
            # Save in traditional pickle format
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'use_scaler': self.use_scaler,
                'hyperparameters': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'random_state': self.random_state
                }
            }, f"{save_path}.pkl")
            print(f"✅ Model saved as {save_path}.pkl (pickle format)")

        else:
            raise ValueError("save_format must be either 'pt' or 'pkl'")

    @classmethod
    def load_model(cls, model_path):
        """
        Load XGBoost model from saved file

        Args:
            model_path: Path to the saved model file (with extension)

        Returns:
            XGBoostClassifierWithSave instance
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_path.endswith('.pt'):
            # Load from PyTorch format
            model_data = torch.load(model_path, map_location='cpu')

            # Create new instance
            instance = cls()
            instance.model = model_data['xgb_model']
            instance.feature_names = model_data['feature_names']
            instance.is_fitted = model_data['model_info']['is_fitted']
            instance.use_scaler = model_data['model_info'].get('use_scaler', False)

            # Load hyperparameters
            hyperparams = model_data['hyperparameters']
            instance.n_estimators = hyperparams['n_estimators']
            instance.max_depth = hyperparams['max_depth']
            instance.learning_rate = hyperparams['learning_rate']
            instance.random_state = hyperparams['random_state']

            # Load scaler if present
            if 'scaler' in model_data:
                instance.scaler = model_data['scaler']

            print(f"✅ Model loaded from {model_path} (PyTorch format)")

        elif model_path.endswith('.pkl'):
            # Load from pickle format
            model_data = joblib.load(model_path)

            # Create new instance
            instance = cls()
            instance.model = model_data['model']
            instance.feature_names = model_data['feature_names']
            instance.scaler = model_data.get('scaler')
            instance.use_scaler = model_data.get('use_scaler', False)
            instance.is_fitted = True

            # Load hyperparameters
            hyperparams = model_data['hyperparameters']
            instance.n_estimators = hyperparams['n_estimators']
            instance.max_depth = hyperparams['max_depth']
            instance.learning_rate = hyperparams['learning_rate']
            instance.random_state = hyperparams['random_state']

            print(f"✅ Model loaded from {model_path} (pickle format)")

        else:
            raise ValueError("Model file must be either .pt or .pkl format")

        return instance

    def predict_novel_data(self, novel_features_df):
        """
        Make predictions on novel data with proper preprocessing

        Args:
            novel_features_df: DataFrame with the same features as training data

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Validate features
        if self.feature_names is None:
            raise ValueError("Feature names not available for validation")

        # Check if all required features are present
        missing_features = set(self.feature_names) - set(novel_features_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Reorder columns to match training data
        X_novel = novel_features_df[self.feature_names].copy()

        # Apply scaling if used during training
        if self.use_scaler and self.scaler is not None:
            X_novel = self.scaler.transform(X_novel)

        # Make predictions
        predictions = self.predict(X_novel)
        probabilities = self.predict_proba(X_novel)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'probabilities_ai': probabilities[:, 1],  # Probability of AI-generated
            'probabilities_human': probabilities[:, 0]  # Probability of human-written
        }