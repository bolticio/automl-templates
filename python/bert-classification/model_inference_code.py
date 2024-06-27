import kserve
import torch
import os
from google.cloud import storage
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict

class MedicalConditionBERTModel(kserve.KFModel):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.local_model_dir = "/mnt/models"  # Local directory to save model files
        self.load()

    def download_model_from_gcs(self, gcs_path, local_path):
        client = storage.Client()
        bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        os.makedirs(local_path, exist_ok=True)
        for blob in blobs:
            file_path = os.path.join(local_path, blob.name.replace(prefix, "").lstrip("/"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            blob.download_to_filename(file_path)
        return local_path

    def load(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model_path = self.download_model_from_gcs(self.model_dir, self.local_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(local_model_path)
        self.model = BertForSequenceClassification.from_pretrained(local_model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]
        texts = [instance['text'] for instance in inputs]
        encodings = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    model_dir =os.getenv("GCS_STORAGE")
    model = MedicalConditionBERTModel("medical-condition-bert", model_dir=model_dir)
    kserve.KFServer().start([model])
