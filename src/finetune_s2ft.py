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

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)

from utils.utils import (
    print_rank_0,
    to_device,
    set_random_seed,
    get_all_reduce_mean,
    int_or_float,
)

from utils.s2_utils import (
    convert_ffn_layer_to_s2,
    convert_mha_layer_to_s2,
    convert_s2_to_linear_layer,
    only_optimize_s2_parameters,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    create_hf_model,
    save_hf_format,
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
    print_throughput
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
        "--mlp_criterion",
        type=str,
        default='random',
    )

    parser.add_argument(
        "--adapter_name",
        type=str,
        default='none',
        help="sparse fine-tuning tuner",
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
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
    
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        dropout=args.dropout,
    )

    if args.s2:
        print_rank_0("------use S2FT------", args.global_rank)
        if args.v_ratio > 0 or args.o_ratio > 0:
            parameters_v = {}
            parameters_o = {}
            mha_indices = [
                i
                for i in range(
                    model.config.num_attention_heads * model.config.num_hidden_layers
                )
            ]
            for i in range(model.config.num_hidden_layers):
                parameters_v[i] = []
                parameters_o[i] = []
            num_v = int(
                model.config.num_attention_heads
                * model.config.num_hidden_layers
                * args.v_ratio
            )
            num_o = int(
                model.config.num_attention_heads
                * model.config.num_hidden_layers
                * args.o_ratio
            )
            select_v = sorted(random.sample(mha_indices, num_v))
            for v in select_v:
                parameters_v[v // model.config.num_attention_heads].append(
                    v % model.config.num_attention_heads
                )
            select_o = sorted(random.sample(mha_indices, num_o))
            for o in select_o:
                parameters_o[o // model.config.num_attention_heads].append(
                    o % model.config.num_attention_heads
                )
            selected_parameters_mha = {"v_proj": parameters_v, "o_proj": parameters_o}

            convert_mha_layer_to_s2(model, selected_parameters_mha)

        if args.u_ratio > 0 or args.d_ratio > 0:
            parameters_u = {}
            parameters_d = {}
            intermediate_dim = model.config.intermediate_size
            if args.mlp_criterion == 'random':
                ffn_indices = [
                    i for i in range(intermediate_dim * model.config.num_hidden_layers)
                ]
                for i in range(model.config.num_hidden_layers):
                    parameters_u[i] = []
                    parameters_d[i] = []
                num_u = int(intermediate_dim * model.config.num_hidden_layers * args.u_ratio)
                num_d = int(intermediate_dim * model.config.num_hidden_layers * args.d_ratio)
                select_u = sorted(random.sample(ffn_indices, num_u))
                for u in select_u:
                    parameters_u[u // intermediate_dim].append(u % intermediate_dim)
                select_d = sorted(random.sample(ffn_indices, num_d))
                for d in select_d:
                    parameters_d[d // intermediate_dim].append(d % intermediate_dim)
            
            elif args.mlp_criterion == 'lra':
                filter_rank = args.filter_rank
                for i in range(model.config.num_hidden_layers):
                    parameters_u[i] = []
                    parameters_d[i] = []
                num_u = int(intermediate_dim * args.u_ratio)
                num_d = int(intermediate_dim * args.d_ratio)
                u_count = 0
                d_count = 0
                for name, p in model.named_parameters():
                    if 'up' in name and 'weight' in name and num_u > 0:
                        u, s, v = torch.svd(p.data.to(torch.float32).to(torch.device("cuda")))
                        # filter out the top k singular values
                        if 'hybrid' in args.mask_type:
                            s[filter_rank // 2: -filter_rank // 2] = 0
                        elif 'random' in args.mask_type:
                            random_indices = torch.randperm(s.shape[0])
                            s[random_indices[filter_rank:]] = 0
                        elif 'least' in args.mask_type:
                            s[: -filter_rank] = 0
                        else:
                            s[filter_rank:] = 0
                        # reconstruct the matrix
                        reconstructed = torch.mm(u, torch.mm(torch.diag(s), v.T))
                        assert reconstructed.shape[1] == intermediate_dim, f"{name}, shape: {reconstructed.shape}"
                        up_sum = torch.sum(reconstructed, dim=0)
                        _, top_u = torch.topk(up_sum, k=num_u, dim=0, largest=True, sorted=True)
                        parameters_u[u_count] = top_u.tolist()
                        print(f"up_proj indices: {top_u}")
                        u_count += 1

                    elif 'down' in name and 'weight' in name and num_d > 0:
                        u, s, v = torch.svd(p.data.to(torch.float32).to(torch.device("cuda")))
                        # filter out the top k singular values
                        if 'hybrid' in args.mask_type:
                            s[filter_rank // 2: -filter_rank // 2] = 0
                        elif 'random' in args.mask_type:
                            random_indices = torch.randperm(s.shape[0])
                            s[random_indices[filter_rank:]] = 0
                        elif 'least' in args.mask_type:
                            s[: -filter_rank] = 0
                        else:
                            s[filter_rank:] = 0
                        # reconstruct the matrix
                        reconstructed = torch.mm(u, torch.mm(torch.diag(s), v.T))
                        assert reconstructed.shape[1] == intermediate_dim, f"{name}, shape: {reconstructed.shape}, intermediate: {intermediate_dim}"
                        down_sum = torch.sum(reconstructed, dim=0)
                        _, top_d = torch.topk(down_sum, k=num_d, dim=0, largest=True, sorted=True)
                        parameters_u[d_count] = top_d.tolist()
                        print(f"down_proj indices: {top_d}")
                        d_count += 1

            selected_parameters_ffn = {"up_proj": parameters_u, "down_proj": parameters_d}
            convert_ffn_layer_to_s2(model, selected_parameters_ffn)

        model = only_optimize_s2_parameters(model)
        model = make_model_gradient_checkpointing_compatible(model)

        print_rank_0(f"learning rate: {args.learning_rate}", args.global_rank)
        
    elif args.gradient_checkpointing:
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

    if args.peft_tuner in ['sparse_weight_mag']:
        no_grad_params = ["bias", "layernorm", "embed_tokens", "norm", "lm_head"]
        # no_grad_params = ["bias", "layernorm", "embed_tokens", "norm"]
        # stop model.embed_tokens.weight from being updated
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
    if args.peft_tuner in ['sparse_weight_mag']:
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
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
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
                    accelerator.print(f"Epoch {epoch+1}: Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}")
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")
                            
                            # save_hf_format(unwrapped_model, tokenizer, args)
                
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
            print(f"Validation perplexity: {ppl}, Validation loss: {val_loss}")
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")
                final_saved_model_index = "last"

        model = best_model if best_model is not None else model
        if args.s2:
            print_rank_0("converting s2 to linear layer ...", args.global_rank)
            model = convert_s2_to_linear_layer(model)
        save_hf_format(model, tokenizer, args)


if __name__ == "__main__":
    main()