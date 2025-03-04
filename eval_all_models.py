import os
import subprocess
import argparse
import yaml

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
                # Convert list to a string of comma-separated values (no spaces)
                value_str = "[" + ",".join(map(str, value)) + "]"
                overrides.append(f"{prefix}{key}={value_str}")
            else:
                overrides.append(f"{prefix}{key}={value}")
    generate_overrides(eval_config)
    return overrides

def find_cp_epochs(model_folder):
    """
    Looks for a subfolder named 'cp' in model_folder and returns a list
    of checkpoint epoch values extracted from filenames.
    Expected filename format: model_epoch_<epoch>.pt
    """
    cp_dir = os.path.join(model_folder, "cp")
    epochs = []
    if os.path.exists(cp_dir) and os.path.isdir(cp_dir):
        for file in os.listdir(cp_dir):
            if file.endswith('.pt') and "model_epoch" in file:
                try:
                    # Extract the epoch number.
                    # For example, from 'model_epoch_4000.pt' extract '4000'
                    base = file.split("model_epoch_")[1]
                    epoch = base.split(".pt")[0]
                    epochs.append(epoch)
                except Exception as e:
                    print(f"Failed to extract epoch from {file}: {e}")
    epochs = sorted(epochs, key=int)
    return epochs

def run_eval_for_subfolders(base_folder, eval_config_path):
    # Load evaluation overrides from YAML file.
    overrides = load_eval_overrides(eval_config_path)
    
    # List every subfolder in the base folder (each representing a model directory)
    subfolders = [
        os.path.join(base_folder, name)
        for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name))
    ]
    
    if not subfolders:
        print(f"No subfolders found in {base_folder}")
        return
    
    # Iterate over each model folder
    for model_folder in subfolders:
        epochs = find_cp_epochs(model_folder)
        if not epochs:
            print(f"No checkpoints found in the 'cp' folder of {model_folder}")
            continue
        
        # Create an override that provides a list of checkpoint epochs (comma-separated, no spaces)
        cp_override = f"cp_list=[{','.join(epochs)}]"
        command = [
            "python", "eval_model.py",
            f"model_path={model_folder}",
            cp_override
        ]
        # Append all additional evaluation overrides.
        command.extend(overrides)
        
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run eval_model.py for every model subfolder using an evaluation YAML config."
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
        required=True,
        help="Path to the evaluation YAML config file containing Hydra overrides."
    )
    
    args = parser.parse_args()
    run_eval_for_subfolders(args.base_folder, args.eval_config)
