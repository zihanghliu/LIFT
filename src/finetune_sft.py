import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import torch
import json
import random
import math
import time
import argparse
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    get_scheduler,
)

from utils.utils import (
    print_rank_0,
    get_all_reduce_mean,
    int_or_float,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    save_hf_format,
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

def parse_args():
    parser = argparse.ArgumentParser(description="S2FT Training")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=100,
        help="Size of the validation set. If 0, no validation set is used.",
    )
    parser.add_argument(
        "--load_last_model", action="store_true", help="only save the last model"
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=80,
        help="size of eval_step",
    )
    parser.add_argument(
        "--eval_delay",
        type=int_or_float,
        default=0,
        help="eval after certain steps if it is an integer, or eval after certain ratio of steps if it is a float",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0.,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Training data type",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout rate of the model."
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    # S2FT
    parser.add_argument(
        "--s2", action="store_true", help="use S2FT for efficient training."
    )
    parser.add_argument("--v_ratio", type=float, default=0.0)
    parser.add_argument("--o_ratio", type=float, default=0.0)
    parser.add_argument("--u_ratio", type=float, default=0.0)
    parser.add_argument("--d_ratio", type=float, default=0.0)
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="tensorboard")
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="save deepspeed engine checkpoint for recover the training",
    )

    parser.add_argument(
        "--peft_tuner",
        type=str,
        default='none',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--no_grad",
        type=str,
        default='none',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default='weight_filtered_mag_abs_largest',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--update_interval",
        type=int,
        default=200,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--filter_rank",
        type=int,
        default=128,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--use_flash_attn",
        type=str,
        default='False',
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="sparse fine-tuning tuner",
    )

    parser.add_argument(
        "--adapter_name",
        type=str,
        default='none',
        help="sparse fine-tuning tuner",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)

    args.global_rank = 1
    
    # Load tokenizer and model
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.model_name_or_path:
        model_kwargs = {}
        model_kwargs['torch_dtype'] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            use_flash_attention_2=True if args.use_flash_attn == 'True' else False,
            **model_kwargs
        )
    else:
        # model = AutoModelForCausalLM.from_config(config)
        model = AutoModelForCausalLM.from_config(config)

    if args.gradient_accumulation_steps == 1:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        model.resize_token_embeddings(int(
            8 *
            math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
        
    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # Prepare datasets
    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        
        if args.val_set_size > 0:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
    else:
        raise ValueError("Only json format is supported for now.")

    if 'sparse' in args.peft_tuner:
        layer_dict = {
            'q': 'q_proj',
            'k': 'k_proj',
            'v': 'v_proj',
            'o': 'o_proj',
            'gate': 'gate_proj',
            'up': 'up_proj',
            'down': 'down_proj'
        }
        no_grad_params = ["bias", "layernorm", "norm"]
        for char, val in layer_dict.items():
            if char == 'o':
                char = 'o_'
            if char in args.no_grad:
                no_grad_params.append(val)
        if 'head' in args.no_grad:
            no_grad_params += ['lm_head', "embed_tokens"]
        for name, param in model.named_parameters():
            if any([s in name for s in no_grad_params]):
                param.requires_grad = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(f'param {name} is not trainable')
            pass
        else:
            print(f'param {name} is trainable')

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )

    if args.val_set_size > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        )

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
       args, model, args.weight_decay, args.learning_rate
    )

    ## Init deepspeed optimizer
    if 'sparse' in args.peft_tuner:
        from sparseAdam import SparseAdamW
        optimizer = SparseAdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95), mask_type=args.mask_type
        )
    else:
        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
        )

    

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    print(f"max trainable steps: {max_train_steps}, warmup steps: {args.num_warmup_steps}")
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    args.completed_steps = 0


    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None
    
    # Training function
    def train_epoch(epoch):
        nonlocal best_model, best_eval_loss
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                args.completed_steps += 1
                # if accelerator.is_main_process and step % 100 == 0:
                #     print(f"Epoch {epoch}: Step {step}: Loss {loss.item():.4f}")
                if args.logging_steps and args.completed_steps % args.logging_steps == 0:
                    divisor = args.gradient_accumulation_steps * args.logging_steps
                    avg_loss = accelerator.gather(total_loss).mean().item() / divisor
                    log_vals = [
                        f'LR: {lr_scheduler.get_last_lr()[0]:.8f}',
                        f'Loss: {avg_loss:.6f}',
                    ]
                    if hasattr(model, '_losses'):
                        for k in list(model._losses.keys()):
                            log_vals.append(f'{k}: {model._losses[k] / divisor:.8f}')
                            model._losses[k] = 0
                    log_str = ', '.join(log_vals)
                    print(f"  Step: {args.completed_steps}, {log_str}")
                    accelerator.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        },
                        step=args.completed_steps,
                    )
                    total_loss = 0

                if (
                    args.completed_steps % args.eval_step == 0
                    and args.val_set_size > 0
                    and not args.load_last_model
                ):
                    perplexity, eval_loss = evaluate(model)
                    accelerator.print(f"Epoch {epoch+1} Step {args.completed_steps}: Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")

        return total_loss / len(train_dataloader)

    # Evaluation function
    def evaluate(model):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(args.num_train_epochs):
        train_loss = train_epoch(epoch)
        accelerator.print(f"Epoch {epoch+1}: Average loss = {train_loss:.4f}")

    # Save the final model if no validation was done
    if args.val_set_size == 0 and accelerator.is_main_process and args.output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        
        save_hf_format(unwrapped_model, tokenizer, args)

    if args.output_dir is not None:
        # evaluate last model
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluate(model)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")

        model = best_model if best_model is not None else model
        save_hf_format(model, tokenizer, args)


if __name__ == "__main__":
    main()