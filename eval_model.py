import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from modules.data_processing import load_and_preprocess_data, inverse_scaling
from modules.models import LSTM
from modules import losses
from modules import plot

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def eval_model(cfg: DictConfig):
    model_path = cfg.model_path
    output_folder = os.path.join(model_path, "test")
    os.environ["HYDRA_RUN_DIR"] = output_folder    
    logging.basicConfig(level=logging.INFO)
    
    device = cfg.device
    config_path = os.path.join(model_path, ".hydra", "config.yaml")
    cfg_model = OmegaConf.load(config_path)

    data = load_and_preprocess_data(path=cfg.data_path, scaler=cfg_model.scaling.scaler)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = (
        data["X_train_scaled"], data["X_test_scaled"], data["y_train_scaled"], data["y_test_scaled"], data["y_scaler"]
    )

    y_train = inverse_scaling(y_train_scaled, y_scaler)
    y_test = inverse_scaling(y_test_scaled, y_scaler)

    X_train_scaled = X_train_scaled.float().to(device)
    X_test_scaled = X_test_scaled.float().to(device)
    y_train = y_train.float().to(device)
    y_test = y_test.float().to(device)

    model = LSTM(X_train_scaled.shape[2], cfg_model.model.hidden_size, y_train_scaled.shape[2])
    
    loss_names = cfg.loss_parameters.loss_fn_list
    epochs = cfg.cp_list

    train_losses_dict = {loss_name: [] for loss_name in loss_names}
    test_losses_dict = {loss_name: [] for loss_name in loss_names}

    for cp in epochs:
        cp_path = os.path.join(model_path, "cp", f'model_epoch_{cp}.pt')
        model.load_state_dict(torch.load(cp_path))
        model.to(device)
        model.eval()

        with torch.no_grad():
            y_train_pred = model(X_train_scaled)
            y_test_pred = model(X_test_scaled)
            y_train_pred = inverse_scaling(y_train_pred, y_scaler)
            y_test_pred = inverse_scaling(y_test_pred, y_scaler)

            if cfg.clamp_output_nonnegative:
                y_train_pred = torch.clamp(y_train_pred, min=0)
                y_test_pred = torch.clamp(y_test_pred, min=0)
            
            train_losses_log = []
            test_losses_log = []
            
            for loss_name in loss_names:
                loss_fn = losses.get_loss(loss_name, 
                                          cfg.loss_parameters.tukey_c, 
                                          cfg.loss_parameters.huber_delta, 
                                          cfg.loss_parameters.quantile_quantile)
                train_loss = loss_fn(y_train_pred, y_train).item()
                test_loss = loss_fn(y_test_pred, y_test).item()

                train_losses_dict[loss_name].append(train_loss)
                test_losses_dict[loss_name].append(test_loss)

                train_losses_log.append(f"{loss_name}: {train_loss:.4f}")
                test_losses_log.append(f"{loss_name}: {test_loss:.4f}")
        
        logging.info(f"CP {cp} | Train Losses: {', '.join(train_losses_log)} | Test Losses: {', '.join(test_losses_log)}")
        plot.plot_predictions(y_test.detach().cpu(), y_test_pred.detach().cpu(), i0=0, i1=1000, t_len=y_test_scaled.shape[1], 
                              save_path=os.path.join(output_folder, f"ts_{cp}.pdf"))

    # Plot the individual loss curves.
    for loss_name in loss_names:
        plot.plot_losses(
            train_losses_dict[loss_name], 
            test_losses_dict[loss_name], 
            epochs, 
            loss_name, 
            save_path=os.path.join(output_folder, f"loss_plot_{loss_name}.pdf")
        )
    
    # Write a combined table of losses.
    table_file = os.path.join(output_folder, "losses_table.txt")
    with open(table_file, "w") as f:
        # Create header: first column "epoch_nr", then one column per train and test loss.
        header = ["epoch_nr"]
        for loss_name in loss_names:
            header.append(f"{loss_name}_train")
            header.append(f"{loss_name}_test")
        f.write("\t".join(header) + "\n")
        
        # Write each row.
        for i, cp in enumerate(epochs):
            row = [str(cp)]
            for loss_name in loss_names:
                row.append(f"{train_losses_dict[loss_name][i]:.4f}")
                row.append(f"{test_losses_dict[loss_name][i]:.4f}")
            f.write("\t".join(row) + "\n")
    logging.info(f"Saved combined losses table to {table_file}")

if __name__ == "__main__":
    eval_model()
