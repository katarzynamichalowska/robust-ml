# Robust-ML

Robust machine learning algorithms

## Overview

This repository focuses on developing and implementing robust machine learning algorithms designed to enhance model operation in various real-world scenarios, ensuring resilience against various data perturbations, such as noise, outliers, or data distribution shifts.

## Repository Structure

- **configs/** – Configuration files for training and evaluation.
- **examples/** – Example implementations and use cases.
- **modules/** – Core modules and components of the algorithms.
- **train_model.py** – Script to train the machine learning models.

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

3. **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```

4. **Training the Model**:
   - Prepare your dataset in the required `.npz` format.
   - Configure your training parameters in the `configs/` directory.
   - Run the training script:
     ```bash
     python train_model.py
     ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
