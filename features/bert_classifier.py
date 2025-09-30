import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

class BERTClassifier:
    def __init__(self, model_name='google-bert/bert-base-uncased', max_length=512, epochs=3, lr=1e-3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 2)  # BERT base has 768 hidden size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.classifier.to(self.device)
        self.max_length = max_length
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train, y_train):
        self.model.eval()  # Freeze BERT parameters
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