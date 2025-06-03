# LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning

Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, Shiwei Liu

[Link to Paper](https://arxiv.org/abs/2506.00772)

![system](assets/ICML_LIFT_teaser.png)

## Abstract

Recent studies have shown that supervised fine-tuning of LLMs on a small number of high-quality datasets can yield strong reasoning capabilities. However, full fine-tuning (Full FT), while powerful, is computationally expensive and susceptible to overfitting and catastrophic forgetting, particularly when data is limited. Sparse fine-tuning, which previously achieved notable success by updating only a small subset of model parameters, offers a promising trade-off between efficiency and effectiveness. Yet, it has lagged behind in the LLM era due to the difficulty of identifying parameters truly critical for reasoning. In this work, we state that weights with the largest magnitude after low-rank approximation are critical weights for fine-tuning, which we call *Principal Weights*. Surprisingly, while magnitude-based sparse fine-tuning performs poorly as a baseline on LLM fine-tuning, it becomes highly effective after rank reduction. These insights motivate our method: **L**ow-rank **I**nformed Sparse **F**ine-**T**uning ($\texttt{LIFT}$). $\texttt{LIFT}$ only updates the top 5% *Principal Weights* throughout training and consistently achieves better performance on reasoning tasks than Full FT, while maintaining memory efficiency on par with popular parameter-efficient fine-tuning methods.  In addition to strong performance on target domains such as arithmetic reasoning, $\texttt{LIFT}$ also retains up to 20% more source-domain knowledge, compared to Full FT and LoRA.

### Environment Setup
```bash
# create conda environment and install packages from requirements.txt
conda create -n lift python=3.10
conda activate lift
pip install -r requirements.txt

# install LLM-Adapters repository to obtain datasets
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
```

## Fine-tuning LLaMA-2-7B on MATH-10K dataset
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