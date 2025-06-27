import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses(train_losses, test_losses, epochs, loss_name, save_path=None):
    """
    Plots training and testing losses across epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        test_losses (list): List of testing losses per epoch.
        epochs (list): List of epoch numbers.
        loss_name (str): The name of the loss function (for the title).
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Training loss", marker='o')
    plt.plot(epochs, test_losses, label="Testing loss", marker='s')
    
    plt.xlabel("Epoch")
    plt.ylabel(f"{loss_name.upper()} loss")
    plt.title(f"{loss_name.upper()} over epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_predictions(y_true, y_pred, i0=0, i1=1000, t_len=100, save_path=None):
    """
    Plots actual vs. predicted values with vertical lines every `t_len` samples.
    
    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        i0 (int): Start index for plotting. Default is 0.
        i1 (int): End index for plotting. Default is 1000.
        t_len (int): Interval for vertical dashed lines. Default is 100.
    """
    plt.figure(figsize=(20, 6))
    plt.plot(y_true.flatten()[i0:i1], label='Actual')
    plt.plot(y_pred.flatten()[i0:i1], label='Predicted')
    
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Test Samples and Predictions')

    if t_len > 1:
        for i in range(i0, i1, t_len):
            plt.axvline(i, color='grey', linestyle='--', linewidth=0.5)
    plt.axhline(0.0, color='grey', linestyle='--', linewidth=0.5)

    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_sorted_predictions(y_true, y_pred, point_alpha=0.4, point_size=1):
    """
    Plots sorted ground truth values alongside predictions for better visual comparison.

    Args:
        test_truth (np.ndarray): Ground truth values.
        test_predictions (np.ndarray): Model predictions (unscaled).
        point_alpha (float): Transparency of the scatter plot points. Default is 0.4.
        point_size (int): Size of scatter plot points. Default is 1.

    Returns:
        None
    """
    # Flatten arrays for sorting
    test_truth_flat = y_true.flatten()
    test_predictions_flat = y_pred.flatten()

    # Get sorted indices based on ground truth
    sorted_indices = np.argsort(test_truth_flat)

    # Sort both ground truth and predictions using sorted indices
    sorted_truth = test_truth_flat[sorted_indices]
    sorted_predictions = test_predictions_flat[sorted_indices]

    # Plot sorted ground truth and predictions
    plt.figure(figsize=(20, 6))
    plt.plot(sorted_truth, label='Actual (Sorted)')
    plt.scatter(range(len(sorted_predictions)), sorted_predictions, 
                label='Predicted (Sorted)', alpha=point_alpha, s=point_size, c='darkorange')
    
    plt.xlabel('Sample (Sorted by Ground Truth)')
    plt.ylabel('Value')
    plt.title('Sorted Test Samples and Predictions')
    plt.legend()
    plt.show()

# Example usage:
# plot_sorted_predictions(test_truth, y_test_pred_unscaled)
