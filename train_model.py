import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging

from modules.data_processing import load_and_preprocess_data
from modules.models import LSTM
from modules.training import train_model
import modules.losses as losses
import modules.optimizers as optimizers



@hydra.main(version_base=None, config_path="configs", config_name="config_train")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    model_path = HydraConfig.get().run.dir


    data = load_and_preprocess_data(path=cfg.data.path, scaler=cfg.scaling.scaler)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = (
        data["X_train_scaled"], data["X_test_scaled"], data["y_train_scaled"], data["y_test_scaled"], data["y_scaler"]
    )

    train_loader = DataLoader(TensorDataset(X_train_scaled, y_train_scaled), batch_size=cfg.training.batch_size, shuffle=True)

    model = LSTM(X_train_scaled.shape[2], cfg.model.hidden_size, y_train_scaled.shape[2])
    loss_fn = losses.get_loss(cfg.training.loss_fn, cfg.training.huber_delta, cfg.training.tukey_c, cfg.training.quantile_quantile)
    optimizer = optimizers.get_optimizer(optimizer_name=cfg.training.optimizer, model=model, lr=cfg.training.lr)


    train_loss = train_model(
        model, optimizer, loss_fn, train_loader, cfg.training.nr_epochs, 
        log_freq=cfg.training.log_freq, cp_freq=cfg.training.cp_freq, device=cfg.training.device,
        model_savepath=model_path,
    )

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_scaled)
        test_loss = loss_fn(y_test_pred, y_test_scaled)
        logging.info(f"Test Loss: {test_loss.item()}")

if __name__ == "__main__":
    main()
