import torch.optim as optim

def get_optimizer(optimizer_name, model, lr, eps=1e-8, betas=(0.9, 0.999)):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, eps=eps, betas=betas)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adabelief':
        from adabelief_pytorch import AdaBelief
        return AdaBelief(model.parameters(), lr=lr, eps=eps, betas=betas)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, eps=eps, betas=betas)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized")