"""
Description
"""
import sys
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from modules.data_processing import reshape_into_subseries, rename_duplicate_columns
import plotting
from scipy.stats import trim_mean
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def robust_stats(x):
    """
    Compute robust summary statistics for a Series x.
    Returns a Series with:
      - median: the median value
      - mad: median absolute deviation
      - iqr: interquartile range (75th percentile minus 25th percentile)
      - trimmed_mean: 10% trimmed mean
      - p5: 5th percentile
      - p95: 95th percentile
      - robust_skew: (p95 + p5 - 2*median)/(p95 - p5) if denominator != 0, else 0
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    tmean = trim_mean(x, 0.1)
    p5 = np.percentile(x, 5)
    p95 = np.percentile(x, 95)
    robust_skew = (p95 + p5 - 2 * med) / (p95 - p5) if (p95 - p5) != 0 else 0
    return pd.Series({
        'median': med,
        'mad': mad,
        'iqr': iqr,
        'trimmed_mean': tmean,
        'p5': p5,
        'p95': p95,
        'robust_skew': robust_skew
    })



@hydra.main(version_base=None, config_path=".", config_name="config_data_clustering")
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.data_path, index_col=0)     

    if cfg.time_variable in data.columns:
        data[cfg.time_variable] = pd.to_datetime(data[cfg.time_variable])
    elif data.index.name == cfg.time_variable:
        data.index = pd.to_datetime(data.index)
        data = data.reset_index()
    else:
        raise KeyError(f"'{cfg.time_variable}' not found as a column or index in the data.")
    data["Date"] = data[cfg.time_variable].dt.floor('D')

    if cfg.taglist is not None:
        for sheet in cfg.sheets:
            names_df = pd.read_excel(cfg.taglist, sheet_name=sheet, skiprows=cfg.skiprows_taglist).iloc[:, :3].dropna()
            names_dict = dict(zip(names_df.iloc[:,0], names_df.iloc[:,2]))
            data.rename(columns=names_dict, inplace=True)

    data = rename_duplicate_columns(data)
    targets = cfg.targets
    inputs = cfg.inputs

    if cfg.cluster_sets:

        if len(targets) > 1:
            raise ValueError("Only one target variable is allowed for clustering")
        else:
            target_var = targets[0]

        # Group by 'Date' and compute robust statistics for the target variable.
        robust_daily = data.groupby('Date')[target_var].apply(robust_stats).unstack().reset_index()

        features = ['median', 'mad', 'iqr', 'trimmed_mean', 'p5', 'p95', 'robust_skew']
        X = robust_daily[features].values

        # Standardize the features.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use DBSCAN for clustering (adjust eps and min_samples as needed)
        dbscan = DBSCAN(eps=cfg.clustering.eps, min_samples=cfg.clustering.min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        # Append the cluster labels to the robust_daily DataFrame.
        robust_daily['cluster'] = clusters
        most_numerous = robust_daily['cluster'].value_counts().idxmax()

        # Create a new column 'set' based on whether the cluster equals the most numerous cluster
        robust_daily['set'] = robust_daily['cluster'].apply(lambda x: 'train' if x == most_numerous else 'test')
        data_with_set = data.merge(robust_daily[['Date', 'set']], on='Date', how='left')

        pca = PCA(n_components=2, random_state=cfg.random_state)
        pca_components = pca.fit_transform(X_scaled)

        # Create a DataFrame for PCA results, preserving the Date index.
        pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'], index=robust_daily.index)
        pca_df['set'] = robust_daily['set']

        os.makedirs(cfg.metadata_dir, exist_ok=True)
        plotting.plot_pca(pca_df, os.path.join(cfg.metadata_dir, 'data_clustered_pca_robust_stats.png'))
        plotting.plot_distribution(data_with_set, target_var, os.path.join(cfg.metadata_dir, 'data_clustered_distribution_target.png'))
        plotting.plot_timeline(robust_daily, target_var, os.path.join(cfg.metadata_dir, 'data_clustered_timeline.png'))
        plotting.plot_robust_stats_distributions_grid(robust_daily, features, os.path.join(cfg.metadata_dir, 'data_clustered_stats.png'))
        plotting.plot_timeline_binary(robust_daily, os.path.join(cfg.metadata_dir, 'data_clustered_timeline_binary.png'))

        # Make a table with the cluster counts for each set, as well as robust statistics for each cluster.
        set_counts = robust_daily.groupby('set')['set'].value_counts().astype(int)
        set_stats = robust_daily.groupby(['set'])[features].mean()
        set_stats = pd.concat([set_counts, set_stats], axis=1)
        set_stats.to_csv(os.path.join(cfg.metadata_dir, 'data_clustered_train_test_stats.csv'))

        # Split into X_train and y_train and X_test and y_test based on the 'set' column.
        X_train = data_with_set[data_with_set['set'] == 'train'][inputs].reset_index(drop=True)
        y_train = data_with_set[data_with_set['set'] == 'train'][target_var].reset_index(drop=True)
        X_test = data_with_set[data_with_set['set'] == 'test'][inputs].reset_index(drop=True)
        y_test = data_with_set[data_with_set['set'] == 'test'][target_var].reset_index(drop=True)

        X_train = reshape_into_subseries(X_train, cfg.t_len)
        y_train = reshape_into_subseries(y_train, cfg.t_len)
        X_test = reshape_into_subseries(X_test, cfg.t_len)
        y_test = reshape_into_subseries(y_test, cfg.t_len)

    else:
        data = data[inputs + targets]
        data = reshape_into_subseries(data, t_len=cfg.t_len)

        X_train, X_test, y_train, y_test = train_test_split(
            data[:, :, :-len(targets)],  # Features
            data[:, :, -len(targets)],   # Target
            test_size=cfg.test_size, 
            random_state=cfg.random_state
        )

    y_train = y_train.reshape(-1, cfg.t_len, len(targets))
    y_test = y_test.reshape(-1, cfg.t_len, len(targets))

    for name, array in {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }.items():
        if np.isnan(array).any():
            raise ValueError(f"❌ NaNs found in {name}")
    
    os.makedirs(os.path.dirname(cfg.data_savepath), exist_ok=True)
    np.savez(cfg.data_savepath, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    main()
