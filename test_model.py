import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from modules.data_processing import load_and_preprocess_data
from modules.models import LSTM
from modules import losses
from modules import plot

@hydra.main(version_base=None, config_path="configs", config_name="config_test")
def test_model(cfg: DictConfig):
    model_path = cfg.model_path
    os.environ["HYDRA_RUN_DIR"] = os.path.join(model_path, "test")    
    logging.basicConfig(level=logging.INFO)
    
    device = cfg.device
    config_path = os.path.join(model_path, ".hydra", "config.yaml")
    cfg_model = OmegaConf.load(config_path)

    data = load_and_preprocess_data(path=cfg.data_path, scaler=cfg_model.scaling.scaler)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = (
        data["X_train_scaled"], data["X_test_scaled"], data["y_train_scaled"], data["y_test_scaled"], data["y_scaler"]
    )


    model = LSTM(X_train_scaled.shape[2], cfg_model.model.hidden_size, y_train_scaled.shape[2])
    
    loss_name = cfg.loss_fn
    loss_fn = losses.get_loss(loss_name)

    train_losses = []
    test_losses = []
    epochs = cfg.cp_list

    for cp in epochs:
        # Load model
        model.load_state_dict(torch.load(os.path.join(model_path, "cp", f'model_epoch_{cp}.pt')))
        model.to(device)
        model.eval()

        with torch.no_grad():
            y_train_pred = model(X_train_scaled)
            y_test_pred = model(X_test_scaled)
            train_loss = loss_fn(y_train_pred, y_train_scaled).item()
            test_loss = loss_fn(y_test_pred, y_test_scaled).item()

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            logging.info(f"CP {cp} \t Train Loss: {train_loss:.4f} \t Test Loss: {test_loss:.4f}")

    plot.plot_losses(train_losses, test_losses, epochs, loss_name, os.path.join(model_path, "test"))

  
if __name__ == "__main__":
    test_model()
