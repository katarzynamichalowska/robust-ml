import sys
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from modules.data_processing import reshape_into_subseries, rename_duplicate_columns

@hydra.main(version_base=None, config_path=".", config_name="config_data")
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.data_path, index_col=0)    
    data["Time"] = pd.to_datetime(data["Time"])

    for sheet in cfg.sheets:
        names_df = pd.read_excel(cfg.taglist, sheet_name=sheet, skiprows=2).iloc[:, :3].dropna()
        names_dict = dict(zip(names_df["Navn"], names_df["Beskrivelse"]))
        data.rename(columns=names_dict, inplace=True)

    data = rename_duplicate_columns(data)
    targets = cfg.targets
    inputs = cfg.inputs
    data = data[inputs + targets]
    data = reshape_into_subseries(data, t_len=cfg.t_len)

    X_train, X_test, y_train, y_test = train_test_split(
        data[:, :, :-len(targets)],  # Features
        data[:, :, -len(targets)],   # Target
        test_size=cfg.test_size, 
        random_state=cfg.random_state
    )
    y_train = y_train.reshape(-1, cfg.t_len, len(targets))
    y_test = y_test.reshape(-1, cfg.t_len, len(targets))

    os.makedirs(os.path.dirname(cfg.data_savepath), exist_ok=True)
    np.savez(cfg.data_savepath, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    main()
