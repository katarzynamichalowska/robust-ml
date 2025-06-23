import numpy as np
import torch

def load_and_preprocess_data(path, scaler):
    """
    Load, scale, and convert to tensors. Handles two npz formats.
    """
    data = np.load(path)

    X_train, y_train = data["X_train"], data["y_train"]

    if "X_test" in data and "y_test" in data:
        # Standard format
        X_test = data["X_test"]
        y_test = data["y_test"]

        X_test_scaled, _ = scaling(X_test, scaler)
        y_test_scaled, _ = scaling(y_test, scaler)

        return {
            "X_train_scaled": torch.tensor(scaling(X_train, scaler)[0]).float(),
            "y_train_scaled": torch.tensor(scaling(y_train, scaler)[0]).float(),
            "X_test_scaled": torch.tensor(X_test_scaled).float(),
            "y_test_scaled": torch.tensor(y_test_scaled).float(),
            "X_scaler": scaling(X_train, scaler)[1],
            "y_scaler": scaling(y_train, scaler)[1],
        }

    elif all(k in data for k in ["X_test_in", "y_test_in", "X_test_out", "y_test_out"]):
        # In/Out format
        X_test_in, y_test_in = data["X_test_in"], data["y_test_in"]
        X_test_out, y_test_out = data["X_test_out"], data["y_test_out"]

        X_train_scaled, X_scaler = scaling(X_train, scaler)
        y_train_scaled, y_scaler = scaling(y_train, scaler)

        X_test_in_scaled, _ = scaling(X_test_in, X_scaler)
        y_test_in_scaled, _ = scaling(y_test_in, y_scaler)
        X_test_out_scaled, _ = scaling(X_test_out, X_scaler)
        y_test_out_scaled, _ = scaling(y_test_out, y_scaler)

        return {
            "X_train_scaled": torch.tensor(X_train_scaled).float(),
            "y_train_scaled": torch.tensor(y_train_scaled).float(),
            "X_test_in_scaled": torch.tensor(X_test_in_scaled).float(),
            "y_test_in_scaled": torch.tensor(y_test_in_scaled).float(),
            "X_test_out_scaled": torch.tensor(X_test_out_scaled).float(),
            "y_test_out_scaled": torch.tensor(y_test_out_scaled).float(),
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
        }

    else:
        raise ValueError("Unexpected data format in the .npz file.")


def rename_duplicate_columns(df):
    """
    Renames duplicate columns in a DataFrame by appending a unique numeric suffix.

    Args:
        df (pd.DataFrame): Input DataFrame with potentially duplicate column names.

    Returns:
        pd.DataFrame: DataFrame with unique column names.
    """
    new_columns = []
    seen = {}

    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col} {seen[col]+1}")

    df.columns = new_columns
    return df

# Example usage:
# data = rename_duplicate_columns(data)


def reshape_into_subseries(data, t_len):
    """
    Reshapes the given dataset into subseries of length `t_len`.

    Args:
        data (pd.DataFrame or np.ndarray): Input data to reshape.
        t_len (int): Length of each subseries.

    Returns:
        np.ndarray: Reshaped array of shape (n_chunks, t_len, -1).
    """
    n_chunks = len(data) // t_len
    return data[:n_chunks * t_len].values.reshape(n_chunks, t_len, -1)

# Example usage:
# subset_data_reshaped = reshape_into_subseries(subset_data, t_len)


def scaling(f, scaler):
    """
    Scales the input array for each feature independently.

    Parameters:
    f (numpy.ndarray): Input data of shape (N, T, F).
    scaler (str or dict): The scaler to be used for scaling. Can be one of the following:
        - "standard": Standard scaling using mean and standard deviation.
        - "minmax": Min-max scaling using minimum and maximum values.
        - "norm": Normalization scaling using L2 norm.
        - "robust": Robust scaling using median and interquartile range (IQR).

        If a dictionary is provided, it should contain the following keys:
        - "scaler" (str): The scaler type.
        - Additional keys depending on the scaler type:
            - For "standard" scaler: "mean" (array) and "std" (array).
            - For "minmax" scaler: "min" (array) and "max" (array).
            - For "norm" scaler: "norm" (array).
            - For "robust" scaler: "median" (array) and "iqr" (array).

    Returns:
    tuple: A tuple containing the scaled array and a dictionary with scaler information.

    Raises:
    ValueError: If an invalid scaler type is provided.
    """

    scaler_type = scaler if (isinstance(scaler, str) or (scaler is None)) else scaler['scaler']
    F = f.shape[-1] # Number of features 

    if scaler_type is None:
        f_scaled = f
        scaler_type = None
        scaler_features = dict({})

    elif scaler_type == "standard":
        if isinstance(scaler, str):
            f_mean = f.mean(axis=(0, 1), keepdims=True)  # Compute per-feature mean
            f_std = f.std(axis=(0, 1), keepdims=True)  # Compute per-feature std
        else:
            f_mean, f_std = scaler['mean'], scaler['std']
        f_scaled = (f - f_mean) / f_std
        scaler_features = {"mean": f_mean.squeeze(), "std": f_std.squeeze()}

    elif scaler_type == "minmax":
        if isinstance(scaler, str):
            f_min = f.min(axis=(0, 1), keepdims=True)  # Compute per-feature min
            f_max = f.max(axis=(0, 1), keepdims=True)  # Compute per-feature max
        else:
            f_min, f_max = scaler['min'], scaler['max']

        f_scaled = (f - f_min) / (f_max - f_min)
        scaler_features = {"min": f_min.squeeze(), "max": f_max.squeeze()}

    elif scaler_type == "norm":
        f_norm = np.sqrt(np.mean(f**2, axis=(0, 1), keepdims=True))
        f_scaled = f / f_norm
        scaler_features = {"norm": f_norm.squeeze()}

    elif scaler_type == "robust":
        if isinstance(scaler, str):
            f_median = np.median(f, axis=(0, 1), keepdims=True)  # Per-feature median
            f_iqr = np.percentile(f, 75, axis=(0, 1), keepdims=True) - np.percentile(f, 25, axis=(0, 1), keepdims=True)  # Per-feature IQR
        else:
            f_median, f_iqr = scaler['median'], scaler['iqr']

        f_scaled = (f - f_median) / f_iqr
        scaler_features = {"median": f_median.squeeze(), "iqr": f_iqr.squeeze()}

    else:
        print("ERROR: Scaler must be either None, \"standard\", \"minmax\", \"norm\" or \"robust\".")

    scaler_dict = {"scaler": scaler_type, **scaler_features}

    return f_scaled, scaler_dict


def inverse_scaling(z, scaler):
    """
    Inverses the scaling transformation for each feature independently.

    Parameters:
    z (numpy.ndarray or torch.Tensor): Scaled data of shape (N, T, F).
    scaler (dict): Dictionary containing per-feature scaling values.

    Returns:
    numpy.ndarray or torch.Tensor: Original unscaled data of shape (N, T, F).
    """
    if isinstance(z, torch.Tensor):
        # Convert scaler values to tensors for compatibility
        scaler = {k: torch.tensor(v, dtype=z.dtype, device=z.device) if isinstance(v, (int, float, np.ndarray)) else v for k, v in scaler.items()}

    scaler_type = scaler['scaler']

    if scaler_type is None:
        return z  # No scaling applied

    # Ensure scalers are correctly shaped for broadcasting: (1,1,F)
    def _reshape_scaler(value):
        return np.asarray(value).reshape(1, 1, -1) if isinstance(value, np.ndarray) else value

    if scaler_type == "standard":
        mean = _reshape_scaler(scaler['mean'])
        std = _reshape_scaler(scaler['std'])
        x = (z * std) + mean

    elif scaler_type == "minmax":
        min_val = _reshape_scaler(scaler['min'])
        max_val = _reshape_scaler(scaler['max'])
        x = (z * (max_val - min_val)) + min_val

    elif scaler_type == "norm":
        norm = _reshape_scaler(scaler['norm'])
        x = z * norm

    elif scaler_type == "robust":
        median = _reshape_scaler(scaler['median'])
        iqr = _reshape_scaler(scaler['iqr'])
        x = (z * iqr) + median

    else:
        raise ValueError("ERROR: Scaler must be one of None, 'standard', 'minmax', 'norm', or 'robust'.")

    return x

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
