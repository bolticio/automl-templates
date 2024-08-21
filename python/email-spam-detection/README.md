# Email Spam Detection Model

This folder contains the implementation of the Email Spam Detection Model using Python and TensorFlow 2.14. The model is designed to perform binary classification to identify whether an email is spam or not. This template utilizes the `huggingface` library and provides flexibility for different machine learning environments.

This model aims to classify emails as either spam or not spam using a deep learning approach. It leverages the power of TensorFlow 2.14 to build and train the model. The script email_spam_detection.py is the primary handler for training and inference.

The training script requires at least an `a2-highgpu-1g` (12 vCPU and 85 GB Memory) machine type equipped with 1x NVIDIA TESLA A100 GPU, which is available exclusively in the `us-central1` region.

## Dataset

To use the Email Spam Classification Dataset from Kaggle, follow the steps below to download and prepare the data.

1. Upload your Kaggle API key (kaggle.json) using the Colab interface

    ```python
    import os
    import zipfile
    from zipfile import ZipFile
    from google.colab import files

    # Upload your Kaggle API key (kaggle.json) using the Colab interface
    files.upload()

    # Move the uploaded key to the correct location
    !mkdir ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```

2. Authenticate and download the dataset:

    ```python
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Instantiate the Kaggle API client
    api = KaggleApi()

    api.authenticate()

    # Define the directory path where you want to download the dataset
    download_dir = "/content"

    # Download the dataset into the specified directory
    api.dataset_download_files(dataset="purusinghvi/email-spam-classification-dataset", path=download_dir, unzip=True)
    ```

3. Load and read the data:

    ```python
    import pandas as pd

    # Load and Read the Data
    dataset_path = '/content/combined_data.csv'
    df = pd.read_csv(dataset_path)
    df.head(10)
    ```

## Environment Variables

The model is configurable through several environment variables that define the training parameters:

1. **`num_train_epochs`**: Total number of training epochs to perform.
   - **Type**: Integer
   - **Default**: 1
   - **Range**: 1-10

2. **`per_device_train_batch_size`**: The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.
   - **Type**: Integer
   - **Default**: 64
   - **Range**: 8-256

3. **`per_device_eval_batch_size`**: The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.
   - **Type**: Integer
   - **Default**: 64
   - **Range**: 8-256

4. **`warmup_steps`**: Number of steps used for a linear warmup from 0 to the learning rate. Overrides any effect of `warmup_ratio`.
   - **Type**: Integer
   - **Default**: 100
   - **Range**: 0-1000

5. **`weight_decay`**: The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the `AdamW` optimizer.
   - **Type**: Float
   - **Default**: 0.01
   - **Range**: 0-1

6. **`logging_steps`**: Number of update steps between two logs if `logging_strategy="steps"`.
   - **Type**: Integer
   - **Default**: 10
   - **Range**: 10-1000

7. **`early_stopping_patience`**: Use with `metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls.
   - **Type**: Integer
   - **Default**: 3
   - **Range**: 1-10

## Training

### Install Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

### Set Environment Variables

Load the environment variables in your shell:

```sh
export AIP_MODEL_DIR=./
export MODEL_NAME=email-spam-detector
export GCS_STORAGE=gs://your-bucket/models/email-spam-detector
```

### Run the Training Script

Execute the training script with the appropriate parameters:

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 email_spam_detection.py \
    --datasets '{"training_dataset_1": ["./data/email_spam_detection.csv"]}' \
    --model "${AIP_MODEL_DIR}" \
    --metrics "$(pwd)/metrics.json" \
    --hparams '{"num_train_epochs": 1, "per_device_train_batch_size": 64, "per_device_eval_batch_size": 64, "warmup_steps": 100, "weight_decay": 0.01, "logging_steps": 10, "early_stopping_patience": 3}'
```

## Inference

### Run Inference with KServe

Deploy the model and run the inference script:

```sh
python3 inference.py
```

### Sample Curl for Inference

Use the following curl command to test the model's prediction endpoint:

```sh
curl --location 'http://127.0.0.1:8080/v1/models/email-spam-detector:predict' \
--header 'Content-Type: application/json' \
--data '{
    "instances": ["thanks for all your answers guys i know i should have checked the rsync manual but i would rather get a escapenumber sure answer from one of you this is my current script bin bash rsync avt \\ exclude alpha \\ exclude arm \\ exclude hppa \\ exclude hurd \\ exclude iaescapenumber \\ exclude mescapenumberk \\ exclude mips \\ exclude mipsel \\ exclude multi arch \\ exclude powerpc \\ exclude sescapenumber \\ exclude sh \\ exclude sparc \\ exclude source \\ ftp de debian org debian cd var www mirror debian cd i know loads of excludes for now will include more distros soon from the rsync manual del an alias for delete during delete delete extraneous files from dest dirs delete before receiver deletes before transfer default delete during receiver deletes during xfer not before delete after receiver deletes after transfer not before delete excluded also delete excluded files from dest dirs which delete would you suggest i use thanks again john escapelong on escapenumber escapenumber escapenumber olleg samoylov wrote jonathan escapelong wrote sorry for the banal question my favourite keys for escapenumber stage rsync rsync verbose recursive links hard links times filter '\''r tmp '\'' delete after delay updates source url destination log file olleg samoylov www escapelong org mirror escapelong org rcrack escapelong org ninux org wireless community rome"]
}'
```

### Build Image for Inference

To containerize the model for deployment:

```sh
docker build -t email-spam-detector:v1 .
docker run -d -p 8080:8080 \
  -e MODEL_NAME=email-spam-detector \
  -e GCS_STORAGE=gs://your-bucket/models/email-spam-detector \
  email-spam-detector:v1
```
