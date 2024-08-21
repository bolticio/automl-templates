import logging
import os
from pathlib import Path
from typing import Dict, List

import kserve
import torch
from google.cloud.storage import Client
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBertModel(kserve.Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        logger.info(f"Loading model from {self.model_dir}")

        storage_client = Client()
        bucket_name, blob_name = self.extract_bucket_and_blob_name(gcs_uri=self.model_dir)
        bucket = storage_client.get_bucket(bucket_or_name=bucket_name)
        blobs = bucket.list_blobs(prefix=blob_name)  # Get list of files
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            Path(directory).mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(blob.name)

        self.tokenizer = DistilBertTokenizer.from_pretrained(blob_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(blob_name)
        self.model.to(self.device)
        logger.info("Model loaded successfully")
        self.ready = True

    def extract_bucket_and_blob_name(self, gcs_uri: str):
        """Extracts the bucket name and blob name from a GCS URI."""
        uri_without_gs = gcs_uri.removeprefix('gs://')
        bucket_name, _, blob_name = uri_without_gs.partition('/')
        return bucket_name, blob_name

    def predict(self, payload: Dict[str, List[str]], headers: Dict[str, str] = None) -> dict:
        texts = payload.get("instances", [])
        logger.info(f"Received {len(texts)} instances for prediction")

        # Tokenize the input texts
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        encodings = {key: val.to(self.device) for key, val in encodings.items()}

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

        logger.info("Predictions generated successfully")
        return {"predictions": predictions}


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    model_dir = os.getenv("GCS_STORAGE")
    model = DistilBertModel(name=model_name, model_dir=model_dir)
    model.load()
    kserve.ModelServer(workers=1).start([model])
