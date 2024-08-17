import logging
import os

import torch
from kserve import Model, model_server
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBertModel(Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        logger.info(f"Loading model from {self.model_dir}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        logger.info("Model loaded successfully")

    def predict(self, inputs: dict) -> dict:
        texts = inputs.get("instances", [])
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
    model_name = os.getenv("MODEL_NAME", "distilbert-seq-classifier")
    model_dir = os.getenv("GCS_STORAGE", "gs://fcs-c801ed9d-3a1c-4a48-8cf4-11a94808cd41-0f256eb2-us-central1/c801ed9d-3a1c-4a48-8cf4-11a94808cd41/models/66b9dec8229b56623f7f40c9/v1/training/aiplatform-custom-training-2024-08-13-10:14:11.724/model")
    model = DistilBertModel(name=model_name, model_dir=model_dir)
    model.load()
    model_server.start(models=[model])
