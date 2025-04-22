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
from scipy.stats import trim_mean
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

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

def plot_pca(pca_df: pd.DataFrame, save_path: str) -> None:
    """
    Plots the PCA scatter plot with hue set to 'set'.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='set', data=pca_df, palette='viridis', s=50)
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.title('PCA of robust daily statistics (DBSCAN clustering based train-test split)')
    plt.legend(title='Set', loc='best')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_distribution(data_with_set: pd.DataFrame, target_var: str, save_path: str) -> None:
    """
    Plots the KDE of the target variable for training and testing sets.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data_with_set, x=target_var, hue='set', fill=True, common_norm=False, palette='viridis')
    plt.title("Distribution of " + target_var + " in train and test sets")
    plt.xlabel(target_var)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_timeline(robust_daily: pd.DataFrame, target_var: str, save_path: str) -> None:
    """
    Plots the daily median of the target variable over time colored by train/test set.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(robust_daily['Date'], robust_daily['median'],
                c=robust_daily['set'].map({'train': 'blue', 'test': 'red'}),
                s=50, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Median ' + target_var)
    plt.title('Train-test split based on DBSCAN clustering')
    train_patch = mpatches.Patch(color='blue', label='Train')
    test_patch = mpatches.Patch(color='red', label='Test')
    plt.legend(handles=[train_patch, test_patch], title="Set", loc='best')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_robust_stats_distributions_grid(robust_daily: pd.DataFrame, features: list, save_path: str) -> None:
    """
    Plots KDE distributions of all robust statistics in a 2x4 subplot layout, split by train/test set.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.kdeplot(data=robust_daily, x=feature, hue='set', fill=True, common_norm=False,
                    palette='viridis', ax=axes[i])
        axes[i].set_title(f"{feature}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density")

    # Hide any unused subplot (if features < 8)
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution of robust statistics in train and test sets", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_timeline_binary(robust_daily: pd.DataFrame, save_path: str) -> None:
    """
    Plots a binary timeline of the train-test split: 0 for train, 1 for test.
    """
    plt.figure(figsize=(10, 3))
    y_vals = robust_daily['set'].map({'train': 0, 'test': 1})
    plt.scatter(robust_daily['Date'], y_vals, c=y_vals.map({0: 'blue', 1: 'red'}), s=40, alpha=0.7)
    plt.yticks([0, 1], ['Train', 'Test'])
    plt.xlabel('Date')
    plt.title('Train-test split over time (DBSCAN clustering)')
    train_patch = mpatches.Patch(color='blue', label='Train')
    test_patch = mpatches.Patch(color='red', label='Test')
    plt.legend(handles=[train_patch, test_patch], title="Set", loc='best')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



@hydra.main(version_base=None, config_path=".", config_name="config_data_clustering")
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.data_path, index_col=0) 

    if cfg.time_variable in data.columns:
        data[cfg.time_variable] = pd.to_datetime(data[cfg.time_variable])
    elif data.index.name == cfg.time_variable:
        data.index = pd.to_datetime(data.index)
        data = data.reset_index()  # make time a column
    else:
        raise KeyError(f"'{cfg.time_variable}' not found as a column or index in the data.")
    data["Date"] = data[cfg.time_variable].dt.floor('D')

    if cfg.taglist is not None:
        for sheet in cfg.sheets:
            names_df = pd.read_excel(cfg.taglist, sheet_name=sheet, skiprows=2).iloc[:, :3].dropna()
            names_dict = dict(zip(names_df["Navn"], names_df["Beskrivelse"]))
            data.rename(columns=names_dict, inplace=True)

    data = rename_duplicate_columns(data)
    print(data.columns)
    targets = cfg.targets
    inputs = cfg.inputs

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
    plot_pca(pca_df, os.path.join(cfg.metadata_dir, 'data_clustered_pca_robust_stats.png'))
    plot_distribution(data_with_set, target_var, os.path.join(cfg.metadata_dir, 'data_clustered_distribution_target.png'))
    plot_timeline(robust_daily, target_var, os.path.join(cfg.metadata_dir, 'data_clustered_timeline.png'))
    plot_robust_stats_distributions_grid(robust_daily, features, os.path.join(cfg.metadata_dir, 'data_clustered_stats.png'))
    plot_timeline_binary(robust_daily, os.path.join(cfg.metadata_dir, 'data_clustered_timeline_binary.png'))

    # Make a table with the cluster counts for each set, as well as robust statistics for each cluster.
    set_counts = robust_daily.groupby('set')['set'].value_counts().astype(int)
    set_stats = robust_daily.groupby(['set'])[features].mean()
    set_stats = pd.concat([set_counts, set_stats], axis=1)
    set_stats.to_csv(os.path.join(cfg.metadata_dir, 'data_clustered_train_test_stats.csv'))

    # Split into X_train and y_train and X_test and y_test based on the 'set' column.
    X_train = data_with_set[data_with_set['set'] == 'train'][inputs]
    y_train = data_with_set[data_with_set['set'] == 'train'][target_var]
    X_test = data_with_set[data_with_set['set'] == 'test'][inputs]
    y_test = data_with_set[data_with_set['set'] == 'test'][target_var]

    X_train = reshape_into_subseries(X_train, cfg.t_len)
    y_train = reshape_into_subseries(y_train, cfg.t_len)
    X_test = reshape_into_subseries(X_test, cfg.t_len)
    y_test = reshape_into_subseries(y_test, cfg.t_len)


    # Reshape into 3D arrays for LSTM input.
    y_train = y_train.reshape(-1, cfg.t_len, 1)
    y_test = y_test.reshape(-1, cfg.t_len, 1)

    os.makedirs(os.path.dirname(cfg.data_savepath), exist_ok=True)
    np.savez(cfg.data_savepath, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if __name__ == "__main__":
    main()
