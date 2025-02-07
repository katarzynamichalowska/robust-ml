import numpy as np
import torch

def load_and_preprocess_data(path, scaler):
    """
    Load, scale and convert to tensors.
    """
    data = np.load(path)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    # Scale data
    X_train_scaled, X_scaler = scaling(X_train, scaler)
    y_train_scaled, y_scaler = scaling(y_train, scaler)
    X_test_scaled, _ = scaling(X_test, X_scaler)
    y_test_scaled, _ = scaling(y_test, y_scaler)
    X_train_scaled = torch.tensor(X_train_scaled).float()
    y_train_scaled = torch.tensor(y_train_scaled).float()
    X_test_scaled = torch.tensor(X_test_scaled).float()
    y_test_scaled = torch.tensor(y_test_scaled).float()

    return {
        "X_train_scaled": X_train_scaled,
        "y_train_scaled": y_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_test_scaled": y_test_scaled,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler
    }

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
    Scales the input array f using the specified scaler.

    Parameters:
    f (numpy.ndarray): The input array to be scaled.
    scaler (str or dict): The scaler to be used for scaling. Can be one of the following:
        - "standard": Standard scaling using mean and standard deviation.
        - "minmax": Min-max scaling using minimum and maximum values.
        - "norm": Normalization scaling using L2 norm.

        If a dictionary is provided, it should contain the following keys:
        - "scaler" (str): The scaler type.
        - Additional keys depending on the scaler type:
            - For "standard" scaler: "mean" (float) and "std" (float).
            - For "minmax" scaler: "min" (float) and "max" (float).

    Returns:
    tuple: A tuple containing the scaled array and a dictionary with scaler information.

    Raises:
    ValueError: If an invalid scaler type is provided.

    Examples:
    >>> f = np.array([1, 2, 3, 4, 5])
    >>> scaling(f, "standard")
    (array([-1.41421356, -0.70710678,  0.,  0.70710678,  1.41421356]), {'scaler': 'standard', 'mean': 3.0, 'std': 1.4142135623730951})

    >>> f = np.array([1, 2, 3, 4, 5])
    >>> scaler = {"scaler": "minmax", "min": 1, "max": 5}
    >>> scaling(f, scaler)
    (array([0., 0.25, 0.5, 0.75, 1.]), {'scaler': 'minmax', 'min': 1, 'max': 5})
    """

    scaler_type = scaler if (isinstance(scaler, str) or (scaler is None)) else scaler['scaler']

    if scaler_type is None:
        f_scaled = f
        scaler_type = None
        scaler_features = dict({})

    elif scaler_type == "standard":
        f_mean, f_std = (f.mean(), f.std()) if isinstance(
            scaler, str) else (scaler['mean'], scaler['std'])
        f_scaled = (f - f_mean) / f_std
        scaler_features = dict({"mean": f_mean, "std": f_std})

    elif scaler_type == "minmax":
        f_min, f_max = (f.min(), f.max()) if isinstance(
            scaler, str) else (scaler['min'], scaler['max'])
        f_scaled = (f - f_min) / (f_max - f_min)
        scaler_features = dict({"min": f_min, "max": f_max})

    elif scaler_type == "norm":
        f_norm = np.sqrt(np.mean(f**2, axis=1))
        f_scaled = np.divide(f.T, f_norm).T
        scaler_features = dict({})
    
    elif scaler_type == "robust":
        f_median, f_iqr = (np.median(f, axis=0), np.percentile(f, 75, axis=0) - np.percentile(f, 25, axis=0)) if isinstance(
            scaler, str) else (scaler['median'], scaler['iqr'])
        f_scaled = (f - f_median) / f_iqr
        scaler_features = {"median": f_median, "iqr": f_iqr}

    else:
        print("ERROR: Scaler must be either None, \"standard\", \"minmax\", \"norm\" or \"robust\".")

    # Scaler info as a dictionary
    scaler_dict = dict({"scaler": scaler_type})
    scaler_dict.update(scaler_features)

    return f_scaled, scaler_dict



def inverse_scaling(z, scaler):

    def _inverse_scaling(z, scaler):
        if scaler['scaler'] == "standard":
            x = standard_scaler_inverse(z, scaler['mean'], scaler['std'])

        elif scaler['scaler'] == "minmax":
            x = minmax_scaler_inverse(z, scaler['min'], scaler['max'])

        elif scaler['scaler'] == "None":
            x = z

        return x

    if isinstance(z, (list, tuple)):
        # TODO: reformulate the condition: doesn't work if the output is a full np array / tensor
        x = [_inverse_scaling(z_i, scaler_i) for z_i, scaler_i in zip(z, scaler)]
    else:
        x = _inverse_scaling(z, scaler)

    return x

def minmax_scaler_inverse(z, minimum, maximum):
    x = (z * (maximum - minimum)) + minimum

    return x

def standard_scaler_inverse(z, mean, std):
    x = (z * std) + mean

    return x