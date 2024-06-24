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

Run the training script

```python
python3 train.py --datasets '{"training_dataset_1": "./data/train.csv"}' --model "$(pwd)" --metrics "$(pwd)/metrics.json" --hparams '{"learning_rate": 0.001, "min_delta": 0.0002, "patience": 20, "epochs": 500, "batch_size": 1000}'
```

## Inference

Run the inference script

```python
python3 inference.py
```
