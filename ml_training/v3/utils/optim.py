"""
Sharpness-Aware Minimization (SAM) optimizer.

SAM seeks parameters in uniformly low-loss neighborhoods for better
generalization.  The two-step process:

  Step 1 (ascent):   w_adv = w + rho * grad(L(w)) / ||grad(L(w))||
  Step 2 (descent):  w    = w - lr  * grad(L(w_adv))

By computing the gradient at a worst-case perturbation of the current
weights, SAM finds flatter minima that transfer better to unseen data.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization" (ICLR 2021).  https://arxiv.org/abs/2010.01412

Used for image models where the high-dimensional parameter space
creates many sharp, overfitting-prone local minima.
"""

import torch


class SAM(torch.optim.Optimizer):
    """SAM wrapping any base optimizer (typically Adam)."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights in the direction of the gradient (ascent step)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights, apply base optimizer step (descent step)."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, 'SAM requires closure'
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad)
                .norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2,
        )
        return norm
