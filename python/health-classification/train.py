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
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=json.loads, required=True)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
parser.add_argument("--metrics", type=str, required=True)
parser.add_argument("--hparams", default={}, type=json.loads)
parser.add_argument("--label", default="Medical Condition", type=str)
args = parser.parse_args()

if args.model.startswith("gs://"):
    args.model = Path("/gcs/" + args.model[5:])
    args.model.mkdir(parents=True)

datasets = args.datasets.get("training_dataset_1")
label = args.label.lower()
hparams = args.hparams

# Load data
df_train = pd.read_csv(datasets[0])
df_valid = pd.read_csv(datasets[1])
df_test = pd.read_csv(datasets[2])

print("training datasets:  ", df_train.head())
# # Preprocess the data
X_train = df_train.drop(columns=["Medical Condition"])
y_train = df_train["Medical Condition"]
X_valid = df_valid.drop(columns=["Medical Condition"])
y_valid = df_valid["Medical Condition"]
X_test = df_test.drop(columns=["Medical Condition"])
y_test = df_test["Medical Condition"]

# Define preprocessing steps for numerical and categorical features
numerical_features = ['Age', 'Billing Amount', 'Room Number']
categorical_features = ['Name', 'Gender', 'Blood Type', 'Date of Admission', 'Doctor', 'Hospital',
                        'Insurance Provider', 'Admission Type', 'Discharge Date', 'Medication', 'Test Results']

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

print(y_train.shape)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# Save the model
model.save('health_model.keras')

with open(f"{str(args.model)}/preprocessor.pkl", 'wb') as f:
    pickle.dump(preprocessor, f)

with open(f"{str(args.model)}/label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)


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

tf.saved_model.save(model, args.model)

logging.info(f"Metrics: {metrics_dict}")
with open(args.metrics, "w") as fp:
    json.dump(metrics_dict, fp)