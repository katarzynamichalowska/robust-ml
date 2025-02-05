import hydra
from omegaconf import DictConfig
from modules.load_data_veas import load_data
from modules.data_processing import preprocess_data
from modules.models import LSTM
from modules.training import train_model
import modules.losses as losses
import modules.optimizers as optimizers
from torch.utils.data import DataLoader, TensorDataset
import torch


@hydra.main(config_path="configs", config_name="config_train")
def main(cfg: DictConfig):
    data = load_data(cfg.data.file, cfg.data.taglist, sheets=cfg.data.sheets)
    targets = cfg.training.target
    inputs = cfg.training.inputs
    data = data[inputs + targets]

    data = preprocess_data(data, var_Y=targets, vars_X=inputs, t_len=cfg.training.t_len)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = (
        data["X_train_scaled"], data["X_test_scaled"], data["y_train_scaled"], data["y_test_scaled"], data["y_scaler"]
    )

    model = LSTM(len(inputs), cfg.model.hidden_size, len(targets))
    loss_fn = losses.get_loss(cfg.training.loss_fn, cfg.training.huber_delta, cfg.training.tukey_c, cfg.training.quantile_quantile)
    optimizer = optimizers.get_optimizer(optimizer_name=cfg.training.optimizer, model=model, lr=cfg.training.lr)

    X_train_scaled = torch.tensor(X_train_scaled).float()
    y_train_scaled = torch.tensor(y_train_scaled).float()
    X_test_scaled = torch.tensor(X_test_scaled).float()
    y_test_scaled = torch.tensor(y_test_scaled).float()

    train_loader = DataLoader(TensorDataset(X_train_scaled, y_train_scaled), batch_size=cfg.training.batch_size, shuffle=True)
    train_loss = train_model(
        model, optimizer, loss_fn, train_loader, cfg.training.nr_epochs, 
        log_freq=cfg.training.log_freq, cp_freq=cfg.training.cp_freq, device=cfg.training.device
    )

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_scaled)
        test_loss = loss_fn(y_test_pred, y_test_scaled)
        print(f"Test Loss: {test_loss.item()}")

if __name__ == "__main__":
    main()
