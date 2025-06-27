import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List

from modules.data_processing import load_and_preprocess_data, inverse_scaling
from modules.models import LSTM, MLP
from modules import losses
from modules import plot

def _predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_scaler,
    clamp_nonneg: bool,
) -> torch.Tensor:
    """Forward pass → inverse scaling → optional clamping."""
    with torch.no_grad():
        y_hat = model(x)
        y_hat = inverse_scaling(y_hat, y_scaler)
        if clamp_nonneg:
            y_hat = torch.clamp(y_hat, min=0)
    return y_hat

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def eval_model(cfg: DictConfig):
    model_path = cfg.model_path
    output_dir = os.path.join(model_path, "test")
    os.environ["HYDRA_RUN_DIR"] = output_dir    
    logging.basicConfig(level=logging.INFO)
    
    device = cfg.device
    config_path = os.path.join(model_path, ".hydra", "config.yaml")
    cfg_model = OmegaConf.load(config_path)

    data = load_and_preprocess_data(path=cfg.data_path, scaler=cfg_model.scaling.scaler)
    X_train_scaled = data["X_train_scaled"].float().to(device)
    y_train_scaled, y_scaler = data["y_train_scaled"], data["y_scaler"]
    y_train = inverse_scaling(y_train_scaled, y_scaler).float().to(device)

    has_split = all(k in data for k in (
        "X_test_in_scaled", "y_test_in_scaled",
        "X_test_out_scaled", "y_test_out_scaled",
    ))
    
    if has_split:
        X_test_in_scaled = data["X_test_in_scaled"].float().to(device)
        X_test_out_scaled = data["X_test_out_scaled"].float().to(device)
        y_test_in = inverse_scaling(data["y_test_in_scaled"], y_scaler).float().to(device)
        y_test_out = inverse_scaling(data["y_test_out_scaled"], y_scaler).float().to(device)
    else:
        X_test_scaled = data["X_test_scaled"].float().to(device)
        y_test = inverse_scaling(data["y_test_scaled"], y_scaler).float().to(device)

    

    # Model skeleton
    is_sequence = X_train_scaled.dim() == 3  # True for (B, T, F), False for (B, F)    
    input_dim = X_train_scaled.shape[-1]
    output_dim = y_train_scaled.shape[-1]

    if is_sequence:
        model = LSTM(
            input_size=input_dim,
            hidden_sizes=cfg_model.model.fc_hidden_sizes,
            output_size=output_dim,
            activation=cfg_model.model.activation,
            lstm_hidden_size=cfg_model.model.lstm_hidden_size,
        )
    else:
        model = MLP(
            input_size=input_dim,
            hidden_sizes=cfg_model.model.fc_hidden_sizes,
            output_size=output_dim,
            activation=cfg_model.model.activation
        )

    # Loss bookkeeping    
    loss_names: List[str] = cfg.loss_parameters.loss_fn_list
    epochs: List[int] = sorted(cfg.cp_list)
    valid_epochs: List[int] = []

    train_losses: Dict[str, List[float]] = {ln: [] for ln in loss_names}
    if has_split:
        test_in_losses:  Dict[str, List[float]] = {ln: [] for ln in loss_names}
        test_out_losses: Dict[str, List[float]] = {ln: [] for ln in loss_names}
    else:
        test_losses:     Dict[str, List[float]] = {ln: [] for ln in loss_names}


    # Evaluate the model
    for cp in epochs:
        cp_path = os.path.join(model_path, "cp", f'model_epoch_{cp}.pt')
        if not os.path.exists(cp_path):
            logging.warning("Checkpoint %s not found – skipped.", cp)
            continue
        valid_epochs.append(cp)

        model.load_state_dict(torch.load(cp_path))
        model.to(device).eval()

        # Check for bad weights
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"Parameter {name} contains NaNs or Infs")

        # Inference
        y_train_hat = _predict(model, X_train_scaled, y_scaler, cfg.clamp_output_nonnegative)

        if has_split:
            y_test_hat_in  = _predict(model, X_test_in_scaled,  y_scaler, cfg.clamp_output_nonnegative)
            y_test_hat_out = _predict(model, X_test_out_scaled, y_scaler, cfg.clamp_output_nonnegative)
        else:
            y_test_hat = _predict(model, X_test_scaled, y_scaler, cfg.clamp_output_nonnegative)

        # Losses
        for ln in loss_names:
            fn = losses.get_loss(
                ln,
                cfg.loss_parameters.tukey_c,
                cfg.loss_parameters.huber_delta,
                cfg.loss_parameters.quantile_quantile,
            )

            tr = fn(y_train_hat, y_train).item()
            train_losses[ln].append(tr)

            if has_split:
                te_in  = fn(y_test_hat_in,  y_test_in ).item()
                te_out = fn(y_test_hat_out, y_test_out).item()
                test_in_losses [ln].append(te_in)
                test_out_losses[ln].append(te_out)
            else:
                te = fn(y_test_hat, y_test).item()
                test_losses[ln].append(te)

        # Logging
        tl = ", ".join(f"{ln}: {train_losses[ln][-1]:.4f}" for ln in loss_names)
        if has_split:
            il = ", ".join(f"{ln}: {test_in_losses[ln][-1]:.4f}"  for ln in loss_names)
            ol = ", ".join(f"{ln}: {test_out_losses[ln][-1]:.4f}" for ln in loss_names)
            logging.info("CP %d | TRAIN [%s] | TEST-IN [%s] | TEST-OUT [%s]", cp, tl, il, ol)
        else:
            vl = ", ".join(f"{ln}: {test_losses[ln][-1]:.4f}" for ln in loss_names)
            logging.info("CP %d | TRAIN [%s] | TEST [%s]", cp, tl, vl)

        # Prediction plot
        if has_split:
            plot.plot_predictions(
                y_test_out.cpu(), y_test_hat_out.cpu(),
                i0=0, i1=1_000,
                t_len=y_test_out.shape[1] if y_test_out.dim() == 3 else 1,
                save_path=os.path.join(output_dir, f"ts_{cp}.pdf"),
            )
        else:
            plot.plot_predictions(
                y_test.cpu(), y_test_hat.cpu(),
                i0=0, i1=1_000,
                t_len=y_test.shape[1] if y_test.dim() == 3 else 1,
                save_path=os.path.join(output_dir, f"ts_{cp}.pdf"),
            )


    # Plot the individual loss curves.
    for ln in loss_names:
        if has_split:
            plot.plot_losses(
                train_losses[ln],                  # y-axis curve 1
                test_in_losses[ln],                # y-axis curve 2
                valid_epochs,                            # x-axis
                f"{ln} (IN split)",                # legend / title text
                save_path=os.path.join(output_dir, f"loss_plot_{ln}_in.pdf"),
            )
            plot.plot_losses(
                train_losses[ln],
                test_out_losses[ln],
                valid_epochs,
                f"{ln} (OUT split)",
                save_path=os.path.join(output_dir, f"loss_plot_{ln}_out.pdf"),
            )
        else:
            plot.plot_losses(
                train_losses[ln],
                test_losses[ln],
                valid_epochs,
                ln,
                save_path=os.path.join(output_dir, f"loss_plot_{ln}.pdf"),
            )
    
    # Write a combined table of losses.
    table = os.path.join(output_dir, "losses_table.txt")
    with open(table, "w", encoding="utf-8") as f:
        head = ["epoch"]
        for ln in loss_names:
            head.append(f"{ln}_train")
            if has_split:
                head.extend((f"{ln}_test_in", f"{ln}_test_out"))
            else:
                head.append(f"{ln}_test")
        f.write("\t".join(head) + "\n")

        for i, cp in enumerate(valid_epochs):
            row: List[str] = [str(cp)]
            for ln in loss_names:
                row.append(f"{train_losses[ln][i]:.4f}")
                if has_split:
                    row.append(f"{test_in_losses [ln][i]:.4f}")
                    row.append(f"{test_out_losses[ln][i]:.4f}")
                else:
                    row.append(f"{test_losses[ln][i]:.4f}")
            f.write("\t".join(row) + "\n")

    logging.info("Saved combined loss table to %s", table)

if __name__ == "__main__":
    eval_model()
