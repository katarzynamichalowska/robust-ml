import torch
import torch.optim as optim
#from adabelief_pytorch import AdaBelief
from typing import Union, Iterable, Dict, Any


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
    """
    Sharpness-Aware Minimisation optimiser wrapper (AMP-safe).
    * Works with any base optimiser that follows the SGD/Adam API.
    * Adds:
        - eps : tiny constant in grad-norm denominator
        - grad_clip : optional max-norm to tame huge gradients
    """
    def __init__(
        self,
        params: Union[Iterable, torch.nn.Parameter],
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        eps: float = 1e-12,
        grad_clip: float | None = 1.0,
        **base_kwargs: Dict[str, Any],
    ):
        if rho < 0.0:
            raise ValueError("rho must be non-negative")
        defaults = dict(rho=rho, eps=eps, grad_clip=grad_clip, **base_kwargs)
        super().__init__(params, defaults)

        # Build the real optimiser
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)

    # ------------------------ public interface ---------------------- #
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()                 # FP32 norm

        # ① Skip SAM if norm is tiny or NaN/Inf
        min_norm = 1e-6
        if grad_norm < min_norm or not torch.isfinite(grad_norm):
            if zero_grad:
                self.zero_grad()
            return

        for group in self.param_groups:
            rho      = group["rho"]
            eps      = group["eps"]
            scale_max = 10.0                          # ② max allowed factor
            scale     = torch.clamp(rho / (grad_norm + eps),
                                    max=scale_max)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])     # restore
        self.base_optimizer.step()               # real update

        if zero_grad:
            self.zero_grad()

    def zero_grad(self, *args, **kwargs):
        self.base_optimizer.zero_grad(*args, **kwargs)

    # ------------------------ helper methods ------------------------ #
    def _grad_norm(self) -> torch.Tensor:
        norms = []
        max_norm = self.param_groups[0]["grad_clip"]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().float()      # FP32 for stability
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_([g], max_norm)
                norms.append(torch.norm(g, p=2))
        if not norms:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.stack(norms), p=2)