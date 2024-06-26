import logging
import os
import pickle
from typing import Dict

import kserve
import numpy as np
import pandas as pd
from keras.models import Model
from kserve import Model
from kserve.model import ModelInferRequest
from tensorflow.keras.layers import TFSMLayer


class DefaultCustomModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.bucket_name, self.base_path = self.extract_bucket_and_blob_name(os.getenv("GCS_STORAGE"))
        self.download_many_blobs_with_transfer_manager(bucket_name=self.bucket_name,
                                                       blob_names=[f"{self.base_path}/fingerprint.pb",
                                                                   f"{self.base_path}/saved_model.pb",
                                                                   f"{self.base_path}/variables/variables.data-00000-of-00001",
                                                                   f"{self.base_path}/variables/variables.index",
                                                                   f"{self.base_path}/preprocessor.pkl",
                                                                   f"{self.base_path}/label_encoder.pkl"],
                                                       destination_directory="./")
        self.load()

    def extract_bucket_and_blob_name(self, gcs_uri: str):
        """Extracts the bucket name and blob name from a GCS URI."""
        uri_without_gs = gcs_uri.removeprefix('gs://')
        bucket_name, _, blob_name = uri_without_gs.partition('/')
        return bucket_name, blob_name

    def download_many_blobs_with_transfer_manager(self,
                                                  bucket_name,
                                                  blob_names,
                                                  destination_directory="",
                                                  workers=8
                                                  ):
        """Download blobs in a list by name, concurrently in a process pool.

        The filename of each blob once downloaded is derived from the blob name and
        the `destination_directory `parameter. For complete control of the filename
        of each blob, use transfer_manager.download_many() instead.

        Directories will be created automatically as needed to accommodate blob
        names that include slashes.
        """

        from google.cloud.storage import Client, transfer_manager

        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)

        results = transfer_manager.download_many_to_path(
            bucket, blob_names, destination_directory=destination_directory, max_workers=workers
        )

        for name, result in zip(blob_names, results):
            # The results list is either `None` or an exception for each blob in
            # the input list, in order.

            if isinstance(result, Exception):
                print("Failed to download {} due to exception: {}".format(name, result))
            else:
                print("Downloaded {} to {}.".format(name, destination_directory + name))

    def load(self):

        preprocessor_path = os.path.join(self.base_path, "preprocessor.pkl")
        label_encoder_path = os.path.join(self.base_path, "label_encoder.pkl")

        # Load the model
        self.model = TFSMLayer(self.base_path, call_endpoint='serving_default')

        # Load the preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        # Load the label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        logging.info("Model and preprocessors loaded successfully.")
        self.ready = True

    def predict(self, payload: ModelInferRequest, headers: Dict[str, str] = None):
        # Extract data from the request
        instances = payload["instances"]
        input_data = pd.DataFrame.from_dict(instances)

        # Log the shape of input data
        logging.info(f"Input data shape: {input_data.shape}")

        # Preprocess the input data
        input_data_preprocessed = self.preprocessor.transform(input_data)
        input_data_preprocessed = input_data_preprocessed.toarray()

        # Make predictions
        predictions = self.model(input_data_preprocessed)

        predictions = np.array(predictions.get("output_0"))

        decoded_predictions = self.label_encoder.inverse_transform(predictions)

        flat_predictions = [pred[0] for pred in decoded_predictions]

        return {
            "predictions": flat_predictions,
            "max_indices": np.argmax(predictions, axis=1).tolist(),
            "max_values": np.max(predictions, axis=1).tolist()
        }


if __name__ == "__main__":
    model = DefaultCustomModel(os.getenv("MODEL_NAME"))
    model.load()
    kserve.ModelServer(workers=1).start([model])