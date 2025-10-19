import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os
import json

class DeBERTaClassifier:
    def __init__(self, model_name='microsoft/deberta-v3-base', max_length=512, epochs=3, lr=1e-3):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.classifier.to(self.device)
        self.max_length = max_length
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train, y_train):
        self.model.eval()  # Freeze DeBERTa parameters
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for text, label in tqdm(zip(X_train, y_train), desc=f"Epoch {epoch+1}", total=len(X_train)):
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
                logits = self.classifier(cls_emb)
                loss = self.criterion(logits, torch.tensor([label], device=self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        self.classifier.eval()
        preds = []
        with torch.no_grad():
            for text in X_test:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length).to(self.device)
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_emb)
                pred = torch.argmax(logits, dim=1).item()
                preds.append(pred)
        return preds

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'])
        return accuracy, report

    def save_model(self, path):
        """
        Save the trained classifier and configuration to disk.

        Args:
            path: Path to save the model (e.g., 'models/deberta_classifier.pt')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save classifier state and configuration
        save_dict = {
            'classifier_state_dict': self.classifier.state_dict(),
            'model_name': self.model.config._name_or_path,
            'max_length': self.max_length,
            'epochs': self.epochs,
        }

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a trained classifier from disk.

        Args:
            path: Path to the saved model file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Verify model compatibility
        if checkpoint['model_name'] != self.model.config._name_or_path:
            print(f"Warning: Saved model used '{checkpoint['model_name']}' but current model is '{self.model.config._name_or_path}'")

        # Load classifier weights
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.max_length = checkpoint['max_length']
        self.epochs = checkpoint['epochs']

        print(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, path, model_name='microsoft/deberta-v3-base', lr=1e-3):
        """
        Create a DeBERTaClassifier instance from a saved model file.

        Args:
            path: Path to the saved model file
            model_name: DeBERTa model name (should match the saved model)
            lr: Learning rate (only needed if further training is planned)

        Returns:
            DeBERTaClassifier instance with loaded weights
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        # Load checkpoint to get configuration
        checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Create instance with saved configuration
        instance = cls(
            model_name=checkpoint.get('model_name', model_name),
            max_length=checkpoint['max_length'],
            epochs=checkpoint['epochs'],
            lr=lr
        )

        # Load classifier weights
        instance.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        print(f"Model loaded from {path}")
        return instance
