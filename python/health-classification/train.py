import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Check TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=json.loads, required=True)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
parser.add_argument("--metrics", type=str, required=True)
parser.add_argument("--hparams", default={}, type=json.loads)
parser.add_argument("--label", default="Medical Condition", type=str)
args = parser.parse_args()

# Load data
df_train = pd.read_csv(args.datasets.get("training_dataset_1")[0])
df_valid = pd.read_csv(args.datasets.get("training_dataset_1")[1])
df_test = pd.read_csv(args.datasets.get("training_dataset_1")[2])

# Preprocess the data
X_train = df_train.drop(columns=[args.label])
y_train = df_train[args.label]
X_valid = df_valid.drop(columns=[args.label])
y_valid = df_valid[args.label]
X_test = df_test.drop(columns=[args.label])
y_test = df_test[args.label]

# Define preprocessing steps for numerical and categorical features
numerical_features = ['Age', 'Billing Amount', 'Room Number']
categorical_features = ['Name', 'Gender', 'Blood Type', 'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 'Admission Type', 'Discharge Date', 'Medication', 'Test Results']

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_test = preprocessor.transform(X_test)

# Encode labels
label_encoder = OneHotEncoder(sparse_output=False)
y_train = label_encoder.fit_transform(y_train.values.reshape(-1, 1))
y_valid = label_encoder.transform(y_valid.values.reshape(-1, 1))
y_test = label_encoder.transform(y_test.values.reshape(-1, 1))

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[-1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))

# Save the model
if not os.path.exists(args.model):
    os.makedirs(args.model)

model_saved_path = os.path.join(args.model, "models/3")
# model.save(model_saved_path+ "saved_model.keras")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

test_accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Test Accuracy: {test_accuracy}')

# Calculate metrics
metrics_dict = {
    "problem_type": "classification",
    "accuracy": accuracy_score(y_test_classes, y_pred_classes),
    "precision": precision_score(y_test_classes, y_pred_classes, average='weighted'),
    "recall": recall_score(y_test_classes, y_pred_classes, average='weighted'),
    "f1_score": f1_score(y_test_classes, y_pred_classes, average='weighted'),
    "roc_auc": roc_auc_score(y_test, y_pred, multi_class='ovr')
}

logging.info(f"Save model to: {args.model}")
# Create the directory if it doesn't exist
os.makedirs(model_saved_path, exist_ok=True)

with open(os.path.join(model_saved_path, "metrics.json"), "w") as fp:
    json.dump(metrics_dict, fp)

# Save preprocessor
with open(os.path.join(model_saved_path, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)

# Save label_encoder
with open(os.path.join(model_saved_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Save model using TensorFlow saved_model.save
tf.saved_model.save(model, os.path.join(model_saved_path))

sample_input = X_test[:1]  # Use appropriate sample input
print("sample_input+++++++++", sample_input)
logging.info(f"Metrics: {metrics_dict}")
