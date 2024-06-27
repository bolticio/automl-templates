import argparse
import json
import os
import logging
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=json.loads, required=True)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
parser.add_argument("--metrics", type=str, required=True)
parser.add_argument("--hparams", default={}, type=json.loads)
parser.add_argument("--label", default="medical_specialty", type=str)
args = parser.parse_args()

# Load data
df_train = pd.read_csv(args.datasets.get("training_dataset_1")[0])
df_valid = pd.read_csv(args.datasets.get("training_dataset_1")[1])
df_test = pd.read_csv(args.datasets.get("training_dataset_1")[2])

# Combine multiple fields into a single text input
def combine_fields(row):
    return f"{row['transcription']} {row['description']} {row['sample_name']}"

df_train['text'] = df_train.apply(combine_fields, axis=1)
df_valid['text'] = df_valid.apply(combine_fields, axis=1)
df_test['text'] = df_test.apply(combine_fields, axis=1)

# Check the column names
print("Training Data Columns:", df_train.columns)
print("Validation Data Columns:", df_valid.columns)
print("Test Data Columns:", df_test.columns)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MedicalConditionBERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(MedicalConditionBERTClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def train_model(model, data_loader, optimizer, device, num_epochs):
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Prepare data for training
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Adjust these column names as per your dataset
text_column = 'text'
label_column = 'medical_specialty'

# Convert labels to numerical values
label_map = {label: idx for idx, label in enumerate(df_train[label_column].unique())}
df_train[label_column] = df_train[label_column].map(label_map)
df_valid[label_column] = df_valid[label_column].map(label_map)
df_test[label_column] = df_test[label_column].map(label_map)

train_dataset = TextDataset(df_train[text_column], df_train[label_column], tokenizer, max_len=128)
valid_dataset = TextDataset(df_valid[text_column], df_valid[label_column], tokenizer, max_len=128)
test_dataset = TextDataset(df_test[text_column], df_test[label_column], tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model initialization
num_labels = len(label_map)
classifier = MedicalConditionBERTClassifier(num_labels=num_labels)
optimizer = AdamW(classifier.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training
train_model(classifier, train_loader, optimizer, device, num_epochs=3)

# Evaluation
classifier.eval()
y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = classifier(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        y_scores.extend(probs)

y_test_classes = y_true
y_pred_classes = y_pred

# Calculate metrics
metrics_dict = {
    "problem_type": "classification",
    "accuracy": accuracy_score(y_test_classes, y_pred_classes),
    "precision": precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0),
    "recall": recall_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0),
    "f1_score": f1_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0),
    # "roc_auc": roc_auc_score(y_test_classes, y_scores, multi_class='ovr')
}

logging.info(f"Save model to: {args.model}")
# Create the directory if it doesn't exist
model_saved_path = os.path.join(args.model)
os.makedirs(model_saved_path, exist_ok=True)

with open(args.metrics, "w") as fp:
    json.dump(metrics_dict, fp)


# Save model using Torch save
torch.save(classifier.state_dict(), os.path.join(model_saved_path, "MedicalConditionBERTClassifier.pth"))
