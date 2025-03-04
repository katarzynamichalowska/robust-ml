import os
import subprocess
import argparse
import yaml
import matplotlib.pyplot as plt

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    For example: {'a':{'b':1, 'c':2}, 'd':3} becomes {'a.b':1, 'a.c':2, 'd':3}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def load_eval_overrides(eval_config_path):
    """
    Loads the evaluation YAML config and converts its keys into a list
    of Hydra command-line override strings.
    """
    with open(eval_config_path, "r") as f:
        eval_config = yaml.safe_load(f)

    overrides = []
    def generate_overrides(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                generate_overrides(value, prefix=f"{prefix}{key}.")
            elif isinstance(value, list):
                value_str = "[" + ",".join(map(str, value)) + "]"
                overrides.append(f"{prefix}{key}={value_str}")
            else:
                overrides.append(f"{prefix}{key}={value}")
    generate_overrides(eval_config)
    return overrides

def find_cp_epochs(model_folder):
    """
    Looks for a subfolder named 'cp' in model_folder and returns a sorted list
    of checkpoint epoch values extracted from filenames.
    Expected filename format: model_epoch_<epoch>.pt
    """
    cp_dir = os.path.join(model_folder, "cp")
    epochs = []
    if os.path.exists(cp_dir) and os.path.isdir(cp_dir):
        for file in os.listdir(cp_dir):
            if file.endswith('.pt') and "model_epoch" in file:
                try:
                    base = file.split("model_epoch_")[1]
                    epoch = base.split(".pt")[0]
                    epochs.append(epoch)
                except Exception as e:
                    print(f"Failed to extract epoch from {file}: {e}")
    return sorted(epochs, key=int)

def run_eval_for_subfolders(base_folder, eval_config_path):
    """
    Iterates over model folders in base_folder. For each folder it:
      - Finds the checkpoint epochs from the 'cp' folder.
      - Creates an override like: cp_list=[4000,500,...]
      - Runs eval_model.py with the model_path and cp override (plus additional Hydra overrides).
      - eval_model.py writes the losses table (losses_table.txt) in each model folder.
    """
    overrides = load_eval_overrides(eval_config_path)
    
    subfolders = [
        os.path.join(base_folder, name)
        for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name))
    ]
    
    if not subfolders:
        print(f"No subfolders found in {base_folder}")
        return

    for model_folder in subfolders:
        epochs = find_cp_epochs(model_folder)
        if not epochs:
            print(f"No checkpoints found in the 'cp' folder of {model_folder}")
            continue
        
        cp_override = f"cp_list=[{','.join(epochs)}]"
        command = [
            "python", "eval_model.py",
            f"model_path={model_folder}",
            cp_override
        ]
        command.extend(overrides)
        
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True)

def aggregate_results_and_hparams(base_folder):
    """
    Aggregates losses table and hyperparameters for each model folder.
    
    Assumes:
      - Hyperparameters are stored in <model_folder>/.hydra/config.yaml.
      - Losses table is stored in <model_folder>/losses_table.txt, where the header is:
        epoch_nr <tab> <loss1_train> <tab> <loss1_test> <tab> <loss2_train> <tab> <loss2_test> ...
      
    Returns:
      hparams_dict: mapping model folder -> flattened hyperparameter dictionary.
      results_dict: mapping model folder -> { test_loss_column: { epoch: value } }.
                    For example:
                      results_dict[model_folder]["mse_test"][epoch] = loss_value
    """
    hparams_dict = {}
    results_dict = {}
    subfolders = [
        os.path.join(base_folder, name)
        for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name))
    ]
    
    for model_folder in subfolders:
        # Load hyperparameters.
        config_path = os.path.join(model_folder, ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            print(f"No config found in {model_folder}")
            continue
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        hparams_dict[model_folder] = flatten_dict(config)
        
        # Read losses table from losses_table.txt.
        results_path = os.path.join(model_folder, "test", "losses_table.txt")
        if not os.path.exists(results_path):
            print(f"No losses table found in {model_folder}")
            continue
        with open(results_path, "r") as f:
            lines = f.readlines()
        if not lines:
            print(f"Empty losses table in {model_folder}")
            continue

        header = lines[0].strip().split("\t")
        if len(header) < 3:
            print(f"Not enough columns in header in {model_folder}")
            continue
        
        # Identify indices for test loss columns (those ending with '_test')
        test_loss_indices = {}
        for idx, col in enumerate(header):
            if idx == 0:
                continue  # skip epoch_nr column
            if col.endswith("_test"):
                test_loss_indices[col] = idx

        # Initialize storage for each test loss column.
        model_results = { col: {} for col in test_loss_indices }
        
        # Process each line (skip header).
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            try:
                epoch = int(parts[0])
            except Exception as e:
                print(f"Error parsing epoch in {model_folder}: {line} | {e}")
                continue
            for col, idx in test_loss_indices.items():
                try:
                    loss_value = float(parts[idx])
                    model_results[col][epoch] = loss_value
                except Exception as e:
                    print(f"Error parsing {col} value in {model_folder} at epoch {epoch}: {line} | {e}")
                    continue
        
        results_dict[model_folder] = model_results
        
    return hparams_dict, results_dict

def compute_common_hparams(hparams_dict):
    """
    Computes which hyperparameters have the same values across all models.
    Returns a dictionary with the key-value pairs that are common.
    """
    common = {}
    models = list(hparams_dict.keys())
    if not models:
        return common
    first = hparams_dict[models[0]]
    for key, value in first.items():
        common_all = True
        for model in models[1:]:
            if key not in hparams_dict[model] or hparams_dict[model][key] != value:
                common_all = False
                break
        if common_all:
            common[key] = value
    return common

def generate_model_label(model_folder, hparams, common_hparams):
    """
    Generates a label string for a model by including only the hyperparameters
    that differ from the common values.
    """
    label_parts = []
    for key, value in hparams.items():
        if key not in common_hparams:
            label_parts.append(f"{key}={value}")
    if not label_parts:
        return os.path.basename(model_folder)
    return ", ".join(label_parts)

def create_master_plots(results_dict, hparams_dict, savename=None):
    """
    Creates one plot for each test loss column. For each plot, models are plotted with
    test loss vs. epoch, and the curve for each model is labeled with its distinguishing hyperparameters.
    """
    # First, gather the union of test loss column names across all models.
    all_test_loss_cols = set()
    for model_folder, losses in results_dict.items():
        all_test_loss_cols.update(losses.keys())
    all_test_loss_cols = sorted(all_test_loss_cols)
    
    common_hparams = compute_common_hparams(hparams_dict)
    
    for loss_col in all_test_loss_cols:
        plt.figure(figsize=(10, 6))
        for model_folder, losses in results_dict.items():
            if loss_col not in losses:
                continue
            # Get epochs and corresponding test loss values.
            epochs = sorted(losses[loss_col].keys())
            loss_values = [losses[loss_col][ep] for ep in epochs]
            label = generate_model_label(model_folder, hparams_dict.get(model_folder, {}), common_hparams)
            plt.plot(epochs, loss_values, marker='o', label=label)
            
        plt.xlabel("Epoch")
        plt.ylabel(loss_col)
        plt.title(f"{loss_col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if savename:
            plt.savefig(f"{savename}_{loss_col}.pdf")
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run eval_model.py for every model subfolder, aggregate test losses from losses_table.txt, and produce plots for each loss."
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        required=True,
        help="Base folder containing subfolders with model directories."
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        required=False,
        help="Path to the evaluation YAML config file containing Hydra overrides."
    )
    
    args = parser.parse_args()
    
    if args.eval_config:
        run_eval_for_subfolders(args.base_folder, args.eval_config)
    
    hparams_dict, results_dict = aggregate_results_and_hparams(args.base_folder)
    
    if not results_dict:
        print("No results found to plot.")
    else:
        create_master_plots(results_dict, hparams_dict, savename=os.path.basename(os.path.normpath(args.base_folder)))
