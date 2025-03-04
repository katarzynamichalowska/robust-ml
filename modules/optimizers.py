import torch.optim as optim
#from adabelief_pytorch import AdaBelief


def get_optimizer(optimizer_name, model, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-4):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, eps=eps, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':  # Added AdamW
        return optim.AdamW(model.parameters(), lr=lr, eps=eps, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    #elif optimizer_name == 'adabelief':
    #    return AdaBelief(model.parameters(), lr=lr, eps=eps, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, eps=eps, momentum=betas[0], weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized")