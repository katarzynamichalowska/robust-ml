import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd

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