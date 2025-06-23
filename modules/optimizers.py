import torch
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
    

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        super().__init__(self.param_groups, {})  # ← this line fixes the error

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for group in self.param_groups
            for p in group["params"] if p.grad is not None
        ]), p=2)
        return norm
