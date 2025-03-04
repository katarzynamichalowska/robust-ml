import torch
import torch.nn as nn

def get_loss(loss_name, delta=1.0, c=4.685, quantile=0.5):
    if loss_name == 'mse':
        return MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'huber':
        return HuberLoss(delta=delta)
    elif loss_name == 'tukey':
        return TukeyLoss(c=c)
    elif loss_name == 'quantile':
        return QuantileLoss(quantile=quantile)
    else:
        raise ValueError(f"Loss {loss_name} not recognized")

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        """    
        Args:
            delta (float): Threshold for switching between MSE and L1 loss. Default is 1.0.
        """
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)

        loss = 0.5 * error**2 * (abs_error < self.delta) + \
               self.delta * (abs_error - 0.5 * self.delta) * (abs_error >= self.delta)
        
        return loss.mean()

class TukeyLoss(nn.Module):
    def __init__(self, c=4.685):
        """
        Args:
            c (float): Threshold parameter (higher = less sensitivity to outliers, default 4.685).
        """
            
        super(TukeyLoss, self).__init__()
        self.c = c

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        condition = torch.abs(error) < self.c
        loss = torch.where(condition, (self.c**2 / 6) * (1 - (1 - (error / self.c) ** 2) ** 3), self.c**2 / 6)
        return loss.mean()
    
class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        """
        Args:
            quantile (float): Value between (0,1), e.g. 0.25 for 25th percentile or 0.75 for 75th percentile.
        """
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        loss = torch.maximum(self.quantile * error, (self.quantile - 1) * error)  # Vectorized computation
        return torch.mean(loss)