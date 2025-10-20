import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
import os
import json

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.texts[idx], self.labels[idx]
        return self.texts[idx]

class BERTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='google-bert/bert-base-uncased', max_length=512, epochs=3, lr=1e-3, batch_size=16):
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 2)  # BERT base has 768 hidden size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.classifier.to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def get_params(self, deep=True):
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'epochs': self.epochs,
            'lr': self.lr,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reinitialize if needed, but for simplicity, assume params set before fit
        return self

    def collate_fn(self, batch):
        if isinstance(batch[0], tuple):  # Training: (text, label)
            texts, labels = zip(*batch)
            inputs = self.tokenizer(list(texts), return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(self.device)
            labels = torch.tensor(labels, device=self.device)
            return inputs, labels
        else:  # Prediction: text only
            texts = batch
            inputs = self.tokenizer(list(texts), return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(self.device)
            return inputs

    def fit(self, X_train, y_train):
        self.model.eval()  # Freeze BERT parameters
        dataset = TextDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                with torch.no_grad():
                    outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
                logits = self.classifier(cls_emb)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.classes_ = np.array([0, 1])

    def predict(self, X_test):
        self.model.eval()
        self.classifier.eval()
        dataset = TextDataset(X_test)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        preds = []
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_emb)
                batch_preds = torch.argmax(logits, dim=1).tolist()
                preds.extend(batch_preds)
        return preds

    def predict_proba(self, X_test):
        self.model.eval()
        self.classifier.eval()
        dataset = TextDataset(X_test)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        probs = []
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_emb)
                batch_probs = torch.softmax(logits, dim=1).tolist()
                probs.extend(batch_probs)
        return np.array(probs)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate F1 scores directly for full precision
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        report = classification_report(y_test, y_pred, target_names=['Human Written', 'AI Generated'], digits=4)
        return accuracy, report, f1_macro, f1_weighted

    def save_model(self, path):
        """
        Save the trained classifier and configuration to disk.

        Args:
            path: Path to save the model (e.g., 'models/bert_classifier.pt')
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save classifier state and configuration
        save_dict = {
            'classifier_state_dict': self.classifier.state_dict(),
            'model_name': self.model.config._name_or_path,
            'max_length': self.max_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
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

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Verify model compatibility
        if checkpoint['model_name'] != self.model.config._name_or_path:
            print(f"Warning: Saved model used '{checkpoint['model_name']}' but current model is '{self.model.config._name_or_path}'")

        # Load classifier weights
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.max_length = checkpoint['max_length']
        self.epochs = checkpoint['epochs']
        self.batch_size = checkpoint.get('batch_size', 16)
        self.batch_size = checkpoint.get('batch_size', 16)

        print(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, path, model_name='google-bert/bert-base-uncased', lr=1e-3):
        """
        Create a BERTClassifier instance from a saved model file.

        Args:
            path: Path to the saved model file
            model_name: BERT model name (should match the saved model)
            lr: Learning rate (only needed if further training is planned)

        Returns:
            BERTClassifier instance with loaded weights
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        # Load checkpoint to get configuration
        checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True)

        # Create instance with saved configuration
        instance = cls(
            model_name=checkpoint.get('model_name', model_name),
            max_length=checkpoint['max_length'],
            epochs=checkpoint['epochs'],
            lr=lr,
            batch_size=checkpoint.get('batch_size', 16)
        )

        # Load classifier weights
        instance.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        print(f"Model loaded from {path}")
        return instance