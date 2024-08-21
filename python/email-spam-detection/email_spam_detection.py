import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from google.cloud.storage import Client, transfer_manager
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", default={"training_dataset_1": ["./data/email_spam_detection.csv"]}, type=json.loads)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str)
parser.add_argument("--metrics", default=f"{os.getcwd()}/metrics.json", type=str)
parser.add_argument("--hparams", default={"num_train_epochs": 1,
                                          "per_device_train_batch_size": 64,
                                          "per_device_eval_batch_size": 64,
                                          "warmup_steps": 100,
                                          "weight_decay": 0.01,
                                          "logging_steps": 10,
                                          "early_stopping_patience": 3
                                          }, type=json.loads)
args = parser.parse_args()

# Set the local directory for saving model artifacts
local_model_dir = "model"
os.makedirs(local_model_dir, exist_ok=True)

hparams = args.hparams
num_train_epochs = hparams.get("num_train_epochs")
per_device_train_batch_size = hparams.get("per_device_train_batch_size")
per_device_eval_batch_size = hparams.get("per_device_eval_batch_size")
warmup_steps = hparams.get("warmup_steps")
weight_decay = hparams.get("weight_decay")
logging_steps = hparams.get("logging_steps")
early_stopping_patience = hparams.get("early_stopping_patience")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(args.datasets.get("training_dataset_1")[0])
df["text"] = df["text"].apply(lambda x: x.lower())

train_texts, test_texts, train_labels, test_labels = train_test_split(df["text"], df["label"], test_size=0.2)

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move the model to the device
model.to(device)

# Convert texts to input IDs
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)


class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Convert our data into torch Dataset
train_dataset = EmailDataset(train_encodings, train_labels.tolist())
test_dataset = EmailDataset(test_encodings, test_labels.tolist())

training_args = TrainingArguments(
    output_dir=local_model_dir,           # output directory
    num_train_epochs=num_train_epochs,    # total number of training epochs
    per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=per_device_eval_batch_size,   # batch size for evaluation
    warmup_steps=warmup_steps,            # number of warmup steps for learning rate scheduler
    weight_decay=weight_decay,            # strength of weight decay
    logging_dir=f"{local_model_dir}/logs",  # directory for storing logs
    logging_steps=logging_steps,
    eval_strategy="epoch",                # evaluation is performed at the end of each epoch
    save_strategy="epoch",                # save model at the end of each epoch
    fp16=True if device.type == "cuda" else False,  # mixed precision training if GPU is available
    load_best_model_at_end=True           # Load the best model found during training at the end of training
)

# Create the Trainer and train the model
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]  # early stopping callback
)

# Train the model
trainer.train()

# Save the trained model locally
trainer.save_model(local_model_dir)
tokenizer.save_pretrained(local_model_dir)


# Upload local model directory to GCS
def upload_directory_with_transfer_manager(bucket_name: str, source_directory: str, blob_name_prefix: str, workers=8):
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")
    file_paths = [path for path in paths if path.is_file()]
    relative_paths = [path.relative_to(source_directory) for path in file_paths]
    filenames = [str(path) for path in relative_paths]

    print("Found {} files.".format(len(filenames)))

    results = transfer_manager.upload_many_from_filenames(bucket=bucket,
                                                          filenames=filenames,
                                                          source_directory=source_directory,
                                                          blob_name_prefix=blob_name_prefix,
                                                          max_workers=workers)

    for name, result in zip(filenames, results):
        if isinstance(result, Exception):
            print(f"Failed to upload {name} due to exception: {result}")
        else:
            print(f"Uploaded {name} to {bucket.name}/{blob_name_prefix}{name}.")


# Upload model to GCS if args.model starts with gs://
if args.model.startswith("gs://"):
    # Extract bucket name and destination prefix from the GCS path
    bucket_name = args.model[5:].split("/")[0]
    blob_name_prefix = "/".join(args.model[5:].split("/")[1:])

    # Upload the local model directory to GCS
    upload_directory_with_transfer_manager(bucket_name=bucket_name,
                                           source_directory=local_model_dir,
                                           blob_name_prefix=blob_name_prefix)

# Save the training metrics
metrics = trainer.state.log_history[-1]  # Retrieve the latest metrics
with open(args.metrics, "w") as fp:
    json.dump(metrics, fp)
