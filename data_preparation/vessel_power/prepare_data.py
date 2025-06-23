"""
This script downloads and preprocesses the vessel power consumption dataset presented in the following paper:
Malinin, Andrey et al. (2022). Shifts 2.0: Extending The Dataset of Real Distributional Shifts, arXiv:2206.15407.

Info from https://shifts.grand-challenge.org/datasets/:
"The Shifts vessel power estimation dataset consists of measurements sampled every minute from sensors on-board a merchant ship over a span of 4 years, cleaned and augmented with weather data from a third-party provider."
"The task is to predict the ships main engine shaft power, which can be used to predict fuel consumption given an engine model, from the vessel's speed, draft, time since last dry dock cleaning and various weather and sea conditions."

We use the canonical splits proposed by authors:
- Training set (X_train, y_train) corresponds to the training and validation splits in the paper (train.csv, in-distribution)
- Testing set in-distribution (X_val, y_val) corresponds to the validation split in the paper (dev_in.csv, in-distribution) 
- Testing set out-of-distribution (X_test, y_test) corresponds to the test split in the paper (dev_out.csv, out-of-distribution)

Variables in the dataset (verify, this is not the original source):
    Y: 
    - power: the main engine shaft power in kW

    X:
    - draft_aft_telegram: aft draft from the telegram
    - draft_fore_telegram: fore draft from the telegram
    - stw: speed through water
    - diff_speed_overground: difference between the speed over ground and the speed through water
    - awind_vcomp_provider: wind speed from the provider
    - awind_ucomp_provider: wind direction from the provider
    - rcurrent_vcomp: current speed from the vessel
    - rcurrent_ucomp: current direction from the vessel
    - comb_wind_swell_wave_height: combined wind and swell wave height
    - timeSinceDryDock: time since last dry dock cleaning
    - time_id: time of the measurement
"""

import requests
import zipfile 
import io
import pandas as pd
import numpy as np
import os
import hydra
from omegaconf import DictConfig
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

@hydra.main(version_base=None, config_path=".", config_name="config_canonical_split")
def main(cfg: DictConfig):
    url = "https://zenodo.org/record/7684813/files/power_consumption_upload.zip?download=1"
    response = requests.get(url, stream=True)

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    def read_csv_from_zip(zip_file, path):
        with zip_file.open(path) as f:
            return pd.read_csv(f)

    # Loading only the REAL data
    train = read_csv_from_zip(zip_file, "power_consumption_upload/real_data/train.csv")
    dev_in = read_csv_from_zip(zip_file, "power_consumption_upload/real_data/dev_in.csv")
    dev_out = read_csv_from_zip(zip_file, "power_consumption_upload/real_data/dev_out.csv")

    target_col = "power"
    feature_cols = [col for col in train.columns if col != target_col]

    X_train = train[feature_cols].reset_index(drop=True)
    y_train = train[target_col].reset_index(drop=True).to_numpy()[:, np.newaxis]
    X_test_in = dev_in[feature_cols].reset_index(drop=True)
    y_test_in = dev_in[target_col].reset_index(drop=True).to_numpy()[:, np.newaxis]
    X_test_out = dev_out[feature_cols].reset_index(drop=True)
    y_test_out = dev_out[target_col].reset_index(drop=True).to_numpy()[:, np.newaxis]

    # --- Optional: remove time_id from features before modeling ---
    X_train = X_train.drop(columns=["time_id"])
    X_test_in = X_test_in.drop(columns=["time_id"])
    X_test_out = X_test_out.drop(columns=["time_id"])

    os.makedirs(os.path.dirname(cfg.data_savepath), exist_ok=True)
    np.savez(cfg.data_savepath, X_train=X_train, y_train=y_train, X_test_in=X_test_in, y_test_in=y_test_in, X_test_out=X_test_out, y_test_out=y_test_out)
    print(f"Data saved to {cfg.data_savepath}")

if __name__ == "__main__":
    main()
