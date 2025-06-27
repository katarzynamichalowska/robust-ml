import os
import time
import logging
import torch
from torch.amp import autocast, GradScaler


logger = logging.getLogger(__name__)


def train_model(model, optimizer, loss_fn, train_loader, num_epochs, log_freq=10, cp_freq=100, device='cpu', model_savepath=None, use_sam=False, use_amp=False):
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
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(init_scale=2**10)  
    losses = []
    if not os.path.exists("cp"):
        os.makedirs(os.path.join(model_savepath, "cp"))

    for epoch in range(1, num_epochs + 1):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"Param {name} has NaNs or Infs at epoch {epoch}")

        t1 = time.time()
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)


            if use_sam:
                if use_amp:
                    loss_value = sam_step_amp(model, optimizer, loss_fn, batch_X, batch_y, scaler, device)
                else:
                    loss_value = sam_step(model, optimizer, loss_fn, batch_X, batch_y)
            else:
                optimizer.zero_grad()
                if use_amp:
                    with autocast(device_type=device):
                        y_pred = model(batch_X)
                        loss = loss_fn(y_pred, batch_y)
                else:
                    y_pred = model(batch_X)
                    loss = loss_fn(y_pred, batch_y)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                loss_value = loss.item()
            
            epoch_loss += loss_value
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        t2 = time.time()
        
        if epoch % log_freq == 0:
            logger.info(f"Epoch {epoch}, Time: {t2-t1:.2f}s, Loss: {avg_loss:.6f}")
        
        if epoch % cp_freq == 0:
            if model_savepath is not None:
                torch.save(model.state_dict(), os.path.join(model_savepath, "cp", f'model_epoch_{epoch}.pt'))
    return losses


def sam_step(model, optimizer, loss_fn, x, y):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    y_pred = model(x)
    loss_second = loss_fn(y_pred, y)
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    return loss.item()

def sam_step_amp(model, optimizer, loss_fn, x, y, scaler, device,
                 rho_clip=1.0, eps=1e-12):
    optimizer.zero_grad(set_to_none=True)

    # ---- first pass (scaled) ---------------------------------------
    with autocast(device_type=device):
        loss1 = loss_fn(model(x), y)
    scaler.scale(loss1).backward()

    # unscale once -> grads are FP32
    scaler.unscale_(optimizer)

    # 1) clip huge grads
    if rho_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), rho_clip)

    # 2) do first SAM step (the optimiser code must add "eps" to grad_norm)
    optimizer.first_step(zero_grad=True)

    # ---- second pass (un-scaled) -----------------------------------
    with autocast(device_type=device):
        loss2 = loss_fn(model(x), y)
    loss2.backward()                       # no scaler.scale

    # (optional) clip again if you like:
    # torch.nn.utils.clip_grad_norm_(model.parameters(), rho_clip)

    optimizer.second_step(zero_grad=True)

    scaler.update()
    return loss1.item()

