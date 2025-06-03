# LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning

Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, Shiwei Liu

[Link to Paper](https://arxiv.org/abs/2506.00772)

![system](assets/ICML_LIFT_teaser.png)

## Abstract

Recent studies have shown that supervised fine-tuning of LLMs on a small number of high-quality datasets can yield strong reasoning capabilities. However, full fine-tuning (Full FT), while powerful, is computationally expensive and susceptible to overfitting and catastrophic forgetting, particularly when data is limited. Sparse fine-tuning, which previously achieved notable success by updating only a small subset of model parameters, offers a promising trade-off between efficiency and effectiveness. Yet, it has lagged behind in the LLM era due to the difficulty of identifying parameters truly critical for reasoning. In this work, we state that weights with the largest magnitude after low-rank approximation are critical weights for fine-tuning, which we call *Principal Weights*. Surprisingly, while magnitude-based sparse fine-tuning performs poorly as a baseline on LLM fine-tuning, it becomes highly effective after rank reduction. These insights motivate our method: **L**ow-rank **I**nformed Sparse **F**ine-**T**uning (**LIFT**). **LIFT** only updates the top 5% *Principal Weights* throughout training and consistently achieves better performance on reasoning tasks than Full FT, while maintaining memory efficiency on par with popular parameter-efficient fine-tuning methods.  In addition to strong performance on target domains such as arithmetic reasoning, **LIFT** also retains up to 20% more source-domain knowledge, compared to Full FT and LoRA.

## Update
- [x] (6.2.2025) We released the code for LIFT on arithmetic reasoning tasks.

## Installation
```bash
# create conda environment and install packages from requirements.txt
conda create -n lift python=3.10
conda activate lift
pip install -r requirements.txt

# install LLM-Adapters repository to obtain datasets
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
```

## Usage

### General Usage
The `LIFT` method is implemented in `src/sparseAdam.py`, which is a `torch.optim.Optimizer` object. To use it, first create optimizer states, and set three hyperparameters: `lora_rank`, `filter_rank`, and `update_interval`. Then, initialize the `LIFT` object just like initializing Adam.

```python
from src.sparseAdam import SparseAdamW

# initialize optimizer states
no_decay_name_list = [] # customize parameter names that doesn't perform weight decay
weights_with_mask = []
decay_ids = []
other_params_w_decay = []
other_params = []

for module_name, module in model.named_modules():
    if isinstance(module, nn.Linear) and "lm_head" not in module_name and module.weight.requires_grad:
        weights_with_mask.append(module.weight)
        decay_ids.extend([id(module.weight)])

for name, param in model.named_parameters():
    if id(param) not in decay_ids and not any(nd in name for nd in no_decay_name_list) and param.requires_grad:
        other_params_w_decay.append(param)
    elif any(nd in name for nd in no_decay_name_list) and param.requires_grad:
        other_params.append(param)

# set hyperparameters
lora_rank = ...  # rank for low-rank approximation, recommended to be 128
filter_rank = ...  # rank for low rank approximation, recommended to be 128
update_interval = ...  # update interval for the projection matrix

optimizer_grouped_parameters = [
    {
        "params": weights_with_mask,
        "weight_decay": 0,
        "rank": lora_rank,
        "filter_rank": filter_rank,
        "update_proj_gap": update_interval,
        "group_name": "weights_with_mask",
    },
    {
        "params": other_params_w_decay,
        "weight_decay": 0,
        "group_name": "other_params_w_decay",
    },
    {
        "params": other_params,
        "weight_decay": 0.0,
        "group_name": "other_params",
    },
]

# initialize LIFT optimizer
optimizer = SparseAdamW(
    optimizer_grouped_parameters,
    lr=...,  # learning rate
    betas=...,  # beta parameters for Adam
    eps=...,  # epsilon for numerical stability
)
```

### Example: Fine-tuning LLaMA-2-7B on MATH-10K with LIFT
In ```./bash_scripts/finetune_math_lift_llama2_7b.sh```, change the following directories to your actual directories:
```bash
SRC_DIR=/enter/your/path/to/the/repo
DATA_DIR=/enter/your/data/dir
OUTPUT_SRC_DIR=/enter/your/output/dir
```
Then in ```./bash_scripts/eval_math.sh```, change corresponding directories to actual directories:
```bash
SRC_DIR=/enter/your/path/to/the/repo
DATA_DIR=/enter/your/data/dir
```
Then, run the training script:
```bash
bash ./bash_scripts/finetune_math_lift_llama2_7b.sh
```
It will automatically run fine-tuning LLaMA-2-7B on MATH-10K with `LIFT` method, and run evaluation on seven arithmetic reasoning tasks.