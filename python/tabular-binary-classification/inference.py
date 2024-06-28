import os
import pickle
from typing import Dict, Union

import kserve
import numpy as np
import pandas as pd
from kserve.model import ModelInferRequest, ModelInferResponse
from sklearn.preprocessing import MinMaxScaler


class CustomModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.scaler = MinMaxScaler()
        self.ready = False

    def load(self):
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.ready = True

    def preprocess(self, payload: ModelInferRequest,
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

    def predict(self, payload: np.ndarray, headers: Dict[str, str] = None) -> Union[Dict, ModelInferResponse]:
        if payload.size == 0:
            return {}

        predictions = self.model.predict(payload)
        return predictions

    def postprocess(self, result: Union[Dict, ModelInferResponse], headers: Dict[str, str] = None) -> Union[Dict, ModelInferResponse]:
        if not result:
            return result
        binary_predictions = (result >= 0.5).astype(int)

        predictions = ["satisfied" if binary_prediction[0]
                       else "unsatisfied" for binary_prediction in binary_predictions]
        return {"predictions":  predictions}


if __name__ == "__main__":
    model = CustomModel(name=os.getenv("MODEL_NAME", "customer-satisfaction-predictor"))
    model.load()
    kserve.ModelServer(workers=1).start([model])
