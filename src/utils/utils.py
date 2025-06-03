# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import random
import numpy as np
import argparse
from deepspeed.accelerator import get_accelerator

from transformers import set_seed

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)

def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def int_or_float(value):
    try:
        float_val = float(value)
        if float_val.is_integer():
            return int(float_val)
        else:
            return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be an int or a float")