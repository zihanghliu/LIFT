import torch
import tqdm
from transformers import StoppingCriteria

from utils.utils import print_rank_0

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)
    
@torch.no_grad()
def generate_completions(model, device, tokenizer, prompts, batch_size=1, stop_id_sequences=None, disable_tqdm=False, verbose=False, **generation_kwargs):
    generations = []
    if hasattr(model, "module"):
        print_rank_0(f'-----{model.module.generation_config}-----')
    else:
        print_rank_0(f'-----{model.generation_config}-----')

    if generation_kwargs:
        print_rank_0(f'-----{generation_kwargs}-----')
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding = 'longest', return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
            batch_outputs = batch_outputs.detach().cpu()

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # in case piece id out of range
            #batch_outputs[batch_outputs >= tokenizer.vocab_size] = tokenizer.unk_token_id
            #batch_outputs[batch_outputs == -1] = tokenizer.unk_token_id
            
            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if verbose:
            print("--------")
            print(batch_generations[0])
            
        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(batch_input_ids, attention_mask).logits[(torch.arange(batch_input_ids.size(0)), attention_mask.sum(dim=-1)-1)]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs