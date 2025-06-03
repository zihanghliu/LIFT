# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import os
import json
import torch

from transformers import AutoConfig, AutoTokenizer
from utils.utils import print_rank_0

def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for PEFT optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)       # not adding start token 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)      # not adding start token 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})
    return tokenizer

def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    trained=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)
    print_rank_0(f"Creating model {model_class} from {model_name_or_path}")
    if trained:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
       model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)

def save_model_format(model, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)

def get_optimizer_grouped_parameters(
    args,
    model,
    weight_decay,
    learning_rate,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    trainable_name_list=[],
):  
    if 'sparse' in args.peft_tuner:
        import torch.nn as nn
        weights_with_mask = []
        decay_ids = []
        other_params_w_decay = []
        other_params = []
        module_names = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in module_name and module.weight.requires_grad:
                weights_with_mask.append(module.weight)
                module_names.append(module_name)
                assert module.weight.requires_grad
                # get parameter names
                weight_name = module_name + '.weight'
                assert not any(nd in weight_name for nd in no_decay_name_list)
                decay_ids.extend([id(module.weight)])

        for name, param in model.named_parameters():
            if id(param) not in decay_ids and not any(nd in name for nd in no_decay_name_list) and param.requires_grad:
                other_params_w_decay.append(param)
            elif any(nd in name for nd in no_decay_name_list) and param.requires_grad:
                other_params.append(param)
            elif id(param) in decay_ids:
                assert param.requires_grad
            else:
                print(f'param {name} is not trainable')

        optimizer_grouped_parameters = [
            {
                "params": weights_with_mask,
                "weight_decay": 0,
                "rank": args.lora_rank,
                "filter_rank": args.filter_rank,
                "update_proj_gap": args.update_interval,
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
    elif args.adapter_name in ["dora", "lora"]:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (not any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and not any(nd in n.lower() for nd in trainable_name_list))
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and any(nd in n.lower() for nd in trainable_name_list))
                ],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n.lower()
                            for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)

    return non_empty_groups

# This function can be used to print throughput for Step 1 and 2 only
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3

        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config)

        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))

        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )

# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_flops(checkpoint_activations_factor, batch_size, seq_length,
                    hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size *
                           seq_length * num_layers * (hidden_size**2)) * (
                               1.0 + (seq_length / (6.0 * hidden_size)) +
                               (vocab_size /
                                (16.0 * num_layers * hidden_size)))
    return flops_per_iteration


def get_hf_configs(hf_config):
    num_layers = getattr(hf_config, "num_hidden_layers",
                         getattr(hf_config, "n_layer", None))
    hidden_size = getattr(hf_config, "hidden_size",
                          getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size