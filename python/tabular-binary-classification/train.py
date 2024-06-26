import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", default={"training_dataset_1": ["./data/train.csv"]}, type=json.loads)
parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str)
parser.add_argument("--metrics", default=f"{os.getcwd()}/metrics.json", type=str)
parser.add_argument("--hparams", default={
    "learning_rate": 0.001,
    "min_delta": 0.0002,
    "patience": 20,
    "epochs": 500,
    "batch_size": 1000
}, type=json.loads)
args = parser.parse_args()


if args.model.startswith("gs://"):
    args.model = Path("/gcs/" + args.model[5:])
    args.model.mkdir(parents=True)

dataset = args.datasets.get("training_dataset_1")[0]
hparams = args.hparams

train = pd.read_csv(dataset, index_col=0)

features_to_drop = train.nunique()
features_to_drop = features_to_drop.loc[features_to_drop.values == 1].index

# Drop these columns from both the training and the test datasets
train = train.drop(features_to_drop, axis=1)

X = train.iloc[:, :-1]
y = train["TARGET"]


X_resampled, y_resampled = SMOTE().fit_resample(X, y)


# Splitting into train (50%), validation (20%), and test (30%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled,
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=True)

# Further splitting temp into validation (40%) and test (60%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                test_size=0.6,
                                                random_state=42,
                                                shuffle=True)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = keras.Sequential(
    [
        keras.layers.Dense(units=9, activation="relu", input_shape=(X_train.shape[-1],)),
        # Randomly delete 30% of the input units below
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=9, activation="relu"),
        # The output layer, with a single neuron
        keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams.get("learning_rate")),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC()])


early_stopping = EarlyStopping(
    min_delta=hparams.get("min_delta"),  # Minimium amount of change to count as an improvement
    patience=hparams.get("patience"),     # How many epochs to wait before stopping
    restore_best_weights=True)


history = model.fit(X_train, y_train,
                    epochs=hparams.get("epochs"),
                    batch_size=hparams.get("batch_size"),
                    validation_data=(X_val, y_val),
                    verbose=0,
                    # Add in early stopping callback
                    callbacks=[early_stopping])


predictions = model.predict(X_test)
binary_predictions = (predictions >= 0.5).astype(int)

pickle.dump(model, open("/".join([str(args.model), "model.pkl"]), "wb"))

eval_metrics = {
    "accuracy": metrics.accuracy_score(y_test, binary_predictions),
    "average precision score": metrics.average_precision_score(y_test, binary_predictions),
    "classification report": metrics.classification_report(y_test, binary_predictions),
    "confusion_matrix": metrics.confusion_matrix(y_test, binary_predictions).tolist(),
    "roc_auc": metrics.roc_auc_score(y_test, predictions),
    "f1 score": metrics.f1_score(y_test, binary_predictions).tolist(),
    "precision_score": metrics.precision_score(y_test, binary_predictions).tolist(),
    "recall_score": metrics.recall_score(y_test, binary_predictions).tolist()
}

with open(args.metrics, "w") as fp:
    json.dump(eval_metrics, fp)
