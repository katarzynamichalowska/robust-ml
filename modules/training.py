import os
import time
import logging
import torch

logger = logging.getLogger(__name__)


def train_model(model, optimizer, loss_fn, train_loader, num_epochs, log_freq=10, cp_freq=100, device='cpu', model_savepath=None):
    """
    Trains a PyTorch model with configurable logging and checkpointing.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (callable): Loss function.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        num_epochs (int): Number of training epochs.
        log_freq (int): Frequency of loss logging.
        cp_freq (int): Frequency of model checkpoint saving.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    model.to(device)
    losses = []
    if not os.path.exists("cp"):
        os.makedirs(os.path.join(model_savepath, "cp"))

    for epoch in range(1, num_epochs + 1):
        t1 = time.time()
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        t2 = time.time()
        
        if epoch % log_freq == 0:
            logger.info(f"Epoch {epoch}, Time: {t2-t1:.2f}s, Loss: {avg_loss:.6f}")
        
        if epoch % cp_freq == 0:
            if model_savepath is not None:
                torch.save(model.state_dict(), os.path.join(model_savepath, "cp", f'model_epoch_{epoch}.pt'))
    return losses
