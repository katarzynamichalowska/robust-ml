# Robust-ML

Robust machine learning algorithms.

## Overview

This repository contains robust machine learning algorithms designed to enhance model operation in real-world scenarios, ensuring resilience against various data perturbations, such as noise, outliers, or data distribution shifts.

## Repository Structure

- **configs/** – Configuration files for training and evaluation.
- **examples/** – Example implementations and use cases.
- **modules/** – Core modules and components of the algorithms (data processing, models, losses, plotting, etc.).
- **train_model.py** – Script to train the machine learning models.
- **eval_model.py** – Script to evaluate a single model: it loads checkpoints from the model’s `cp/` folder, computes train and test losses for various loss functions, and writes a tab-separated losses table (`losses_table.txt`) in the model’s test folder.
- **eval_all_models.py** – Script that runs eval_model.py on all models in the base_folder, aggregates evaluation results (losses tables and hyperparameters) and generates master plots—one for each test loss function—where curves are labeled by the hyperparameters that differ between models.

## Input Data Format

The training script expects the input data to be provided in an `.npz` file containing the following keys:

- `X_train` – Training features
- `y_train` – Training labels
- `X_test` – Test features
- `y_test` – Test labels

Ensure that your dataset is stored in this format before running the training script.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/katarzynamichalowska/robust-ml.git
   cd robust-ml
   ```

2. **Set Up the Environment**:
   - Ensure you have Python 3.8 or later installed.
   - It's recommended to use a virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
     To deactivate the virtual environment, type:
     ```bash
     deactivate
     ```

3. **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```

4. **Training the Model**:
   - Prepare your dataset in the required `.npz` format.
   - Configure your training parameters in the `configs/config_train.yaml` file.
   - Run the training script:
     ```bash
     python train_model.py
     ```
   - Trained models are stored in dedicated model folders, as defined in the `hydra.run.dir` parameter in the config (e.g., under `models/YYYY-MM-DD/HH-MM-SS`). Each model folder contains a `.hydra` directory with training hyperparameters and a `cp/` folder with checkpoint files. After running the evaluation file, a `test/` folder is also created in this directory.

### Single Model Evaluation

- **eval_model.py**  
  This script:
  - Loads a model from a given folder.
  - Reads the list of checkpoints from the `cp/` folder.
  - Evaluates the model on both the training and testing sets for various loss functions.
  - Saves a tab-separated table (`losses_table.txt`) in the model’s test output folder. This table includes:
    - **First column**: `epoch_nr`
    - **Other columns**: Loss values for training and testing (e.g., `mse_train`, `mse_test`, `huber_train`, `huber_test`, etc.)

### Multiple Models Evaluation and Aggregating Results

- **eval_all_models.py**  
  This script:
  - Iterates over all model subfolders within a specified base folder.
  - For each model folder, it runs `eval_model.py` (which evaluates the model on multiple checkpoints and saves a tab-separated table (`losses_table.txt`) with training and test loss values in the model’s test folder).
  - Aggregates the losses from the `losses_table.txt` files across all model folders.
  - Generates comparison plots (one plot per test loss column) with each curve labeled with hyperparameters that differ among the models. For example, when comparing otherwise identical models trained with different losses, only the training loss functions are displayed as labels.


#### Usage Example

```bash
python eval_all_models.py --base_folder /models/date-of-training --eval_config configs/config_eval_all.yaml
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
