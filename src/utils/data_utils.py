import copy
import logging
import json
import torch
import transformers

from torch.utils.data import Dataset

from dataclasses import dataclass
from typing import Dict, Sequence

from utils.utils import print_rank_0

IGNORE_INDEX = -100

BASE_PROMPT = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}
                
### Response:
"""


BASE_PROMPT_WITH_INPUT = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}
                
### Response:
"""


def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [json.loads(line) for line in file]
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return None


def get_output_or_chosen(example):
    if "output" in example:
        return example["output"]
    elif "chosen" in example:
        return example["chosen"]
    else:
        raise ValueError("double check your data format")


def get_instruction_or_prompt(example):
    if "instruction" in example:
        return example["instruction"]
    elif "prompt" in example:
        return example["prompt"]
    else:
        raise ValueError("double check your data format")


def get_alpaca_prompt(example):
    if "input" in example and example["input"] != "":
        return BASE_PROMPT_WITH_INPUT.format_map(
            {"instruction": example["instruction"], "input": example["input"]}
        )
    else:
        return BASE_PROMPT.format_map({"instruction": example["instruction"]})


def get_output_or_chosen(example):
    if "output" in example:
        return example["output"]
    elif "chosen" in example:
        return example["chosen"]
    elif "answer" in example:
        return example["answer"].split("####")[0].strip()
    elif "Rationale" in example:
        return example["Rationale"]
    elif "rationale" in example:
        return example["rationale"]
    elif "solution" in example:
        return example["solution"]
    else:
        raise ValueError("double check your data format")


def get_instruction_or_prompt(example):
    if "input" in example and example["input"] != "":
        return example["input"]
    elif "instruction" in example:
        return example["instruction"]
    elif "prompt" in example:
        return example["prompt"]
    elif "question" in example:
        return example["question"]
    elif "Problem" in example:
        return example["Problem"]
    elif "problem" in example:
        return example["problem"]
    else:
        raise ValueError("double check your data format")


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(
        strings,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=False,
    )["input_ids"]

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    print_rank_0("-----------------")
    print_rank_0(examples[0])
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        instruction_type: str,
        args,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_json_data(data_path)  # try both formats
        logging.warning("Formatting inputs...")

        # We might want to clean this up, it's a bit messy
        if instruction_type == "single":
            print_rank_0("single-round conversation", args.global_rank)
            if "chat" not in args.model_name_or_path:
                print_rank_0("base model", args.global_rank)
                if "alpaca" in data_path:
                    sources = [get_alpaca_prompt(example) for example in list_data_dict]
                else:
                    sources = [
                        BASE_PROMPT.format_map(
                            {"instruction": get_instruction_or_prompt(example)}
                        )
                        for example in list_data_dict
                    ]
            else:
                print_rank_0("chat model", args.global_rank)
                sources = []
                for example in list_data_dict:
                    chat = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You will be given a user's question, and you need to answer it.",
                        },
                        {"role": "user", "content": get_instruction_or_prompt(example)},
                    ]
                    source = tokenizer.apply_chat_template(chat, tokenize=False)
                    source += " "
                    sources.append(source)

        targets = [
            f"{get_output_or_chosen(example).replace('</s>', '')} {tokenizer.eos_token}"
            for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
