# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
import numpy as np
from transformers.utils.versions import require_version
import torch.optim as optim

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max+1, eta_min, last_epoch)
        self.T_max=T_max
        self.eta_min=eta_min
    def step(self,current_step):
        self.cosine_stepper.step(current_step)

    def get_dr(self,current_step):
        if current_step>=self.T_max:
          return self.eta_min
        self.step(current_step)
        return self.sgd.param_groups[0]['lr']

class SparseAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        mask_type: str = "weight_filtered_mag_abs_largest",
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
   
        super().__init__(params, defaults)
        self.mask_type = mask_type
        self.update_masks()
        print(f"Mask initialized")
        # self.checksparsity()
        self.state['total_step'] = 0
        self.update_proj_gap = 1e8 # initialize it as extremely large value and will be updated in step function

    def update_masks(self):
        for idx, p in enumerate(self.param_groups[0]['params']):
            state = self.state[p]
            beta1, beta2 = self.param_groups[0]["betas"]
            largest = 'largest' in self.mask_type
            lora_rank = self.param_groups[0]['rank']
            sparsity = min((p.shape[0] + p.shape[1]) * lora_rank, p.shape[0] * p.shape[1])
            if 'sparse' in self.mask_type and 'mask' in state and 'exp_avg' in state:
                prev_mask = state['mask'].to(p.device).to(torch.bool)
                prev_exp_avg = torch.zeros_like(p.grad)
                prev_exp_avg_sq = torch.zeros_like(p.grad)

                prev_exp_avg[prev_mask] = state['exp_avg']
                prev_exp_avg_sq[prev_mask] = state['exp_avg_sq']
                del prev_mask

            # update masks
            if "weight_filtered_mag" in self.mask_type:
                # perform svd
                filter_rank = self.param_groups[0]['filter_rank']
                u, s, v = torch.svd(p.data.to(torch.float32).to(torch.device("cuda")))
                # filter out the top k singular values
                if 'hybrid' in self.mask_type:
                    s[filter_rank // 2: -filter_rank // 2] = 0
                elif 'random' in self.mask_type:
                    random_indices = torch.randperm(s.shape[0])
                    s[random_indices[filter_rank:]] = 0
                elif 'least' in self.mask_type:
                    s[: -filter_rank] = 0
                else:
                    s[filter_rank:] = 0
                # reconstruct the matrix
                reconstructed = torch.mm(u, torch.mm(torch.diag(s), v.T))
                if 'abs' in self.mask_type:
                    flattened = torch.abs(reconstructed.data).flatten()
                else:
                    flattened = reconstructed.data.flatten()
                if lora_rank > 0:
                    top_k_indices = torch.topk(flattened, k=sparsity, largest=largest).indices
                    mask = torch.zeros_like(flattened, dtype=torch.bool)
                    mask[top_k_indices] = True
                    state['mask'] = mask.view(p.shape).to('cpu')
                else:
                    state['mask'] = torch.ones_like(p, dtype=torch.bool).to('cpu')

            else:
                if 'block' in self.mask_type:
                    dim_1 = p.shape[0] - lora_rank
                    dim_2 = p.shape[1] - lora_rank

                    # randomly choose starting index for block
                    start_idx_1 = np.random.randint(0, dim_1)
                    start_idx_2 = np.random.randint(0, dim_2)

                    # create mask
                    mask = torch.zeros_like(p, dtype=torch.bool)
                    mask[:, start_idx_2:start_idx_2 + lora_rank] = True
                    mask[start_idx_1:start_idx_1 + lora_rank, :] = True
                else:
                    flattened = torch.rand_like(p).flatten()
                    top_k_indices = torch.topk(flattened, k=sparsity).indices
                    mask = torch.zeros_like(flattened, dtype=torch.bool)
                    mask[top_k_indices] = True
                state['mask'] = mask.view(p.shape)

            # create sparse gradients
            if 'sparse' in self.mask_type and 'mask' in state and 'exp_avg' in state:
                curr_mask = state['mask'].to(p.device)
                state['exp_avg'] = prev_exp_avg[curr_mask]
                state['exp_avg_sq'] = prev_exp_avg_sq[curr_mask]
                del prev_exp_avg, prev_exp_avg_sq, curr_mask

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            if "rank" in group:
                self.update_proj_gap = group["update_proj_gap"]
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                beta1, beta2 = group["betas"]
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                # GaLore Projection
                if "rank" in group:
                    update_mask = state['mask'].to(p.device).bool()
                    if 'sparse' in self.mask_type:
                        grad = grad[update_mask]

                        
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
            
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # # TEST AVERAGE 
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                if "rank" in group:
                    grad=p.grad
                    if 'sparse' in self.mask_type:
                        grad[update_mask]=norm_grad
                    else:
                        if 'adam_mag' in self.mask_type:
                            state['adam_update'] = norm_grad
                        grad[update_mask]=norm_grad[update_mask]
                    grad[~update_mask]=0
                    p.add_(grad, alpha=-step_size)
                
                else:
                    grad=norm_grad
                    p.add_(grad, alpha=-step_size)

                if group["weight_decay"] > 0:
                    if "rank" in group:
                        p.data[state['mask']].add_(p.data[state['mask']],alpha=(-group["lr"] * group["weight_decay"]))
                    else:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        self.state['total_step'] += 1 
        if self.state['total_step'] != 0:
            if (self.state['total_step'] + 1) % self.update_proj_gap == 0:
                if "weight_filtered_mag_act" not in self.mask_type:
                    self.update_masks()
                    print("Mask Update", flush=True)

        return loss
