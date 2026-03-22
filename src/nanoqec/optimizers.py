"""Optimizer helpers for NanoQEC training."""

from __future__ import annotations

import torch


class Lion(torch.optim.Optimizer):
    """A minimal Lion optimizer implementation for local experimentation."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimizer step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(gradient, alpha=1.0 - beta1)
                parameter.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(gradient, alpha=1.0 - beta2)

        return loss
