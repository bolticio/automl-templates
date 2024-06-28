# Tabular binary classification with Neural Networks: Keras

Create a fully-connected artificial neural network for binary classification using [Keras](https://keras.io/), the Python deep learning API.

## Dataset

For the data use the [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) dataset.

You are provided with an anonymized dataset containing a large number of numeric variables. The `TARGET` column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.

The task is to predict the probability that each customer in the test set is an unsatisfied customer.

### File descriptions

1. **train.csv** - the training set including the target
2. **test.csv** - the test set without the target

## Training

Install the required packages.

```python
cd python/tabular-binary-classification
pip3 install -r requirements.txt
```

Load the environment variables in shell.

```sh
export AIP_MODEL_DIR=./
export MODEL_NAME=customer-satisfaction-predictor
export GCS_STORAGE=gs://fcs-c801ed9d-3a1c-4a48-8cf4-11a94808cd41-0f256eb2-asia-south1/models/667d218426ede99100cd83d0/v1/training/aiplatform-custom-training-2024-06-27-08:41:12.011/model
```

Run the training script

```python
python3 train.py --datasets '{"training_dataset_1": ["./data/train.csv"]}' --model "$(pwd)" --metrics "$(pwd)/metrics.json" --hparams '{"learning_rate": 0.001, "min_delta": 0.0002, "patience": 20, "epochs": 500, "batch_size": 1000}'
```

## Testing

Run the training script

```python
python3 test.py
```

## Inference

Run the KServe inference script

```python
python3 inference.py
```

Sample curl for inference.

```curl
curl --location 'http://127.0.0.1:8080/v1/models/customer-satisfaction-predictor:predict' \
--header 'Content-Type: application/json' \
--data '{
    "instances": []
}'
```

Use the `test.json` file for testing instances.
