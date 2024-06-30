import os
import pickle
from typing import Dict, Union

import kserve
import numpy as np
import pandas as pd
from google.cloud.storage import Client
from kserve.model import ModelInferRequest, ModelInferResponse
from sklearn.preprocessing import MinMaxScaler


class CustomModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.scaler = MinMaxScaler()
        self.ready = False
        self.file_name = "model.pkl"
        self.bucket_name = None
        self.blob_name = None

    def extract_bucket_and_blob_name(self, gcs_uri: str):
        """Extracts the bucket name and blob name from a GCS URI."""
        uri_without_gs = gcs_uri.removeprefix('gs://')
        bucket_name, _, blob_name = uri_without_gs.partition('/')
        return bucket_name, blob_name

    def download_blob(self, bucket_name: str, source_blob_name: str, destination_file_name: str):
        """Downloads a blob from the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"

        # The ID of your GCS object
        # source_blob_name = "storage-object-name"

        # The path to which the file should be downloaded
        # destination_file_name = "local/path/to/file"

        storage_client = Client()

        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name}"
              f" to local file {destination_file_name}.")

    def load(self):
        if not os.path.exists(self.file_name):
            self.bucket_name, self.blob_name = self.extract_bucket_and_blob_name(gcs_uri=os.getenv("GCS_STORAGE"))

            self.download_blob(bucket_name=self.bucket_name,
                               source_blob_name=f"{self.blob_name}/{self.file_name}",
                               destination_file_name=self.file_name)

        with open(self.file_name, "rb") as f:
            self.model = pickle.load(f)
        self.ready = True

    async def preprocess(self, payload: ModelInferRequest,
                         headers: Dict[str, str] = None) -> np.ndarray:
        if not payload["instances"]:
            return np.array([])

        features_to_drop = ["ID", "ind_var2_0", "ind_var2", "ind_var27_0", "ind_var28_0", "ind_var28",
                            "ind_var27", "ind_var41", "ind_var46_0", "ind_var46", "num_var27_0",
                            "num_var28_0", "num_var28", "num_var27", "num_var41", "num_var46_0",
                            "num_var46", "saldo_var28", "saldo_var27", "saldo_var41", "saldo_var46",
                            "imp_amort_var18_hace3", "imp_amort_var34_hace3",
                            "imp_reemb_var13_hace3", "imp_reemb_var33_hace3",
                            "imp_trasp_var17_out_hace3", "imp_trasp_var33_out_hace3",
                            "num_var2_0_ult1", "num_var2_ult1", "num_reemb_var13_hace3",
                            "num_reemb_var33_hace3", "num_trasp_var17_out_hace3",
                            "num_trasp_var33_out_hace3", "saldo_var2_ult1",
                            "saldo_medio_var13_medio_hace3"]

        inputs_df = pd.DataFrame.from_dict(payload["instances"])
        data = inputs_df.drop(features_to_drop, axis=1)
        return self.scaler.fit_transform(data)

    async def predict(self, payload: np.ndarray, headers: Dict[str, str] = None) -> Union[Dict, ModelInferResponse]:
        if payload.size == 0:
            return []

        predictions = self.model.predict(payload)
        return predictions

    async def postprocess(self, result: np.ndarray,
                          headers: Dict[str, str] = None) -> Union[Dict, ModelInferResponse]:

        if len(result) == 0:
            return {"predictions": []}
        binary_predictions = (result >= 0.5).astype(int)

        predictions = ["unsatisfied" if binary_prediction[0]
                       else "satisfied" for binary_prediction in binary_predictions]
        return {"predictions":  predictions}


if __name__ == "__main__":
    model = CustomModel(name=os.getenv("MODEL_NAME", "customer-satisfaction-predictor"))
    model.load()
    kserve.ModelServer(workers=1).start([model])
