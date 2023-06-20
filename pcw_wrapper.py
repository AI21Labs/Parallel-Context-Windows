from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt


def combine_past_key_values(past_lst: List[Tuple[Tuple[torch.Tensor]]], longest_window_id: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
    n_layers = len(past_lst[0])
    longest_window = past_lst[longest_window_id]
    all_windows_except_longest = past_lst[:longest_window_id] + past_lst[longest_window_id + 1:]
    return tuple(
        (torch.cat([longest_window[i][0]] + [c[i][0][:, :, 1:, :] for c in all_windows_except_longest], dim=2),
         torch.cat([longest_window[i][1]] + [c[i][1][:, :, 1:, :] for c in all_windows_except_longest], dim=2))
        for i in range(n_layers))


def generate_pcw_position_ids(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]]) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    n_task_tokens = position_ids.shape[1] - sum_windows_size
    position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:  # i.e., first token is already generated
        position_ids = position_ids[:, -1].unsqueeze(-1)
    elif windows_key_values:  # i.e., we are in the first token generation
        position_ids = position_ids[:, sum_windows_size:]
    return position_ids


class PCWModelWrapper:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 device: str,
                 context_window_size: int,
                 right_indentation: bool = False
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.context_window_size = context_window_size
        self.device = device
        # Left indentation is the default behavior as explained in the paper.
        self.right_indentation = right_indentation

    def _get_windows(self, texts: List[str]) -> List[Dict]:
        windows = []
        if self.right_indentation:
            max_window_size = max(n_tokens_in_prompt(self.tokenizer, t, add_special_tokens=True) for t in texts)

        for text in texts:
            encoded_input_window = self.tokenizer(text, return_tensors='pt').to(self.device)
            window_size = encoded_input_window['input_ids'].shape[1]
            if self.right_indentation:
                shift = max_window_size - window_size
                encoded_input_window["position_ids"] = encoded_input_window["attention_mask"].cumsum(-1) - 1 + shift
            with torch.no_grad():
                output = self.model(**encoded_input_window)
            windows.append({'text': text,
                            'encoded_input': encoded_input_window,
                            'attention_mask': encoded_input_window['attention_mask'],
                            'window_size': window_size,
                            'output': output,
                            'past': output['past_key_values']})
        return windows

    def get_contexts_cache(self, contexts: List[str]) -> Dict:
        windows = self._get_windows(contexts)
        windows_sizes = [window['window_size'] for window in windows]
        j = np.argmax(windows_sizes)
        # Windows contain bos tokens, we remove all but one to avoid multiple bos
        return {'past_key_values': combine_past_key_values([window['past'] for window in windows], j),
                'max_window_size': max(windows_sizes),
                'past_attention_mask': torch.cat(
                    [windows[j]['attention_mask']] + [window['attention_mask'][:, 1:] for window in
                                                      windows[:j] + windows[j + 1:]], dim=1),
                'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)}

    def pcw_generate(self,
                     contexts: Optional[List[str]] = None,
                     task_text: Optional[str] = None,
                     contexts_cache: Optional[Dict] = None,
                     restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
                     **kwargs
                     ) -> str:
        """Note: Batching is not supported by PCW at the moment. """
        assert (contexts is None) != (
                contexts_cache is None), "pcw_generate should work with contexts or cache, not with both!"
        cache = contexts_cache or self.get_contexts_cache(contexts)
        encoded_task_text = self.tokenizer(task_text, add_special_tokens=False, return_tensors='pt').to(self.device)
        if restrictive_logit_preprocessor:
            restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_task_text['input_ids'].shape[1])
            kwargs['logits_processor'] = [restrictive_logit_preprocessor]
        combined_attention_mask = torch.cat((cache['past_attention_mask'], encoded_task_text['attention_mask']),
                                            dim=1).to(self.device)
        with torch.no_grad():
            res = self.model.generate(input_ids=encoded_task_text['input_ids'],
                                      attention_mask=combined_attention_mask,
                                      windows_key_values=cache['past_key_values'],
                                      max_window_size=cache['max_window_size'],
                                      sum_windows_size=cache['sum_windows_size'],
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      **kwargs)[0]
        res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
        return self.tokenizer.decode(res[encoded_task_text['input_ids'].shape[1]:])
