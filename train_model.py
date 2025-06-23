import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging

from modules.data_processing import load_and_preprocess_data
from modules.models import MLP, LSTM
from modules.training import train_model
import modules.losses as losses
import modules.optimizers as optimizers


@hydra.main(version_base=None, config_path="configs", config_name="config_train")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    model_path = HydraConfig.get().run.dir


    data = load_and_preprocess_data(path=cfg.data.path, scaler=cfg.scaling.scaler)

    X_train_scaled, y_train_scaled, y_scaler = data["X_train_scaled"], data["y_train_scaled"], data["y_scaler"]

    train_loader = DataLoader(TensorDataset(X_train_scaled, y_train_scaled), batch_size=cfg.training.batch_size, shuffle=True)

    input_dim = X_train_scaled.shape[-1]
    output_dim = y_train_scaled.shape[-1]
    is_sequence = X_train_scaled.dim() == 3  # True for (B, T, F), False for (B, F)

    if is_sequence:
        model = LSTM(
            input_size=input_dim,
            hidden_sizes=cfg.model.fc_hidden_sizes,
            output_size=output_dim,
            activation=cfg.model.activation,
            lstm_hidden_size=cfg.model.lstm_hidden_size,
        )
    else:
        model = MLP(
            input_size=input_dim,
            hidden_sizes=cfg.model.fc_hidden_sizes,
            output_size=output_dim,
            activation=cfg.model.activation
        )

    loss_fn = losses.get_loss(cfg.training.loss_fn, cfg.training.huber_delta, cfg.training.tukey_c, cfg.training.quantile_quantile)
    base_optimizer = optimizers.get_optimizer(
        optimizer_name=cfg.training.optimizer,
        model=model,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    if cfg.training.use_sam:
        optimizer = optimizers.SAM(model.parameters(), base_optimizer=type(base_optimizer), rho=cfg.training.sam_rho, **base_optimizer.defaults)
    else:
        optimizer = base_optimizer

    train_loss = train_model(
        model, optimizer, loss_fn, train_loader, cfg.training.nr_epochs, 
        log_freq=cfg.training.log_freq, cp_freq=cfg.training.cp_freq, device=cfg.training.device,
        model_savepath=model_path, use_sam=cfg.training.use_sam
    )

    model.eval()
    with torch.no_grad():
        if "X_test_scaled" in data:
            # Standard test case
            X_test_scaled = data["X_test_scaled"].to(cfg.training.device)
            y_test_scaled = data["y_test_scaled"].to(cfg.training.device)
            y_test_pred = model(X_test_scaled)
            test_loss = loss_fn(y_test_pred, y_test_scaled)
            print(f"Test loss: {test_loss.item():.4f}")

        elif "X_test_in_scaled" in data and "X_test_out_scaled" in data:
            # In/Out test case
            X_test_in_scaled = data["X_test_in_scaled"].to(cfg.training.device)
            y_test_in_scaled = data["y_test_in_scaled"].to(cfg.training.device)
            X_test_out_scaled = data["X_test_out_scaled"].to(cfg.training.device)
            y_test_out_scaled = data["y_test_out_scaled"].to(cfg.training.device)

            y_pred_in = model(X_test_in_scaled)
            y_pred_out = model(X_test_out_scaled)

            loss_in = loss_fn(y_pred_in, y_test_in_scaled)
            loss_out = loss_fn(y_pred_out, y_test_out_scaled)

            print(f"Test-in loss: {loss_in.item():.4f}")
            print(f"Test-out loss: {loss_out.item():.4f}")

        else:
            raise ValueError("Unexpected test data format for evaluation.")


if __name__ == "__main__":
    main()
