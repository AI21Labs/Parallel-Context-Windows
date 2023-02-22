from typing import List, Tuple, Optional, Dict

import numpy as np
import numpy.typing as npt
import torch
from transformers import GPT2LMHeadModel, LogitsProcessor, PreTrainedTokenizerBase, GPT2Tokenizer
from transformers.configuration_utils import PretrainedConfig

GPT2_WINDOW_SIZE = 1024

LOGIT_BIAS = 100


def combine_past_key_values(past_lst: List[Tuple[Tuple[torch.Tensor]]],
                            contains_bos_token: bool = True) -> Tuple[Tuple[torch.Tensor]]:
    if contains_bos_token:
        # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
        n_layers = len(past_lst[0])
        first_window = past_lst[0]
        return tuple(
            (torch.cat([first_window[i][0]] + [c[i][0][:, :, 1:, :] for c in past_lst[1:]], dim=2),
             torch.cat([first_window[i][1]] + [c[i][1][:, :, 1:, :] for c in past_lst[1:]], dim=2))
            for i in range(n_layers))
    return tuple(
        (torch.cat([c[i][0] for c in past_lst], dim=2), torch.cat([c[i][1] for c in past_lst], dim=2))
        for i in range(len(past_lst[0])))


class RestrictiveTokensLogitsProcessor(LogitsProcessor):
    """ Restrictive decoding is done by adding logits_bias to the relevant tokens. Based on:
    https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability
    """

    def __init__(self,
                 restrictive_token_ids: npt.NDArray[int],
                 eos_token_id: int,
                 prompt_length_to_skip: int = 0,
                 logits_bias: int = LOGIT_BIAS):
        self.restrictive_token_ids = restrictive_token_ids
        self.eos_token_id = eos_token_id
        self.logits_bias = logits_bias
        self.prompt_length_to_skip = prompt_length_to_skip
        self.mask = np.ones(restrictive_token_ids.shape[0], dtype=bool)

        self._preprocess_restrictive_array()

    def _preprocess_restrictive_array(self):
        # extend restrictive_token_ids to include eos as last token for each sequence
        if not (self.restrictive_token_ids[:, -1] == self.eos_token_id).all():
            self.restrictive_token_ids = np.column_stack(
                (self.restrictive_token_ids, np.ones(self.restrictive_token_ids.shape[0]) * self.eos_token_id)).\
                astype(int)

    def update_new_prompt_length_to_skip(self, prompt_length_to_skip: int):
        self.prompt_length_to_skip = prompt_length_to_skip
        self.mask = np.ones(self.restrictive_token_ids.shape[0], dtype=bool)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert input_ids.shape[0] == 1, "This implementation doesn't support batching"
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length > 0:
            self.mask = self.mask & (self.restrictive_token_ids[:, new_tokens_length - 1] == input_ids[
                0, -1].item())
        scores[:, self.restrictive_token_ids[self.mask, new_tokens_length]] += self.logits_bias
        return scores


class GPT2LMHeadWithPCWModel(GPT2LMHeadModel):
    def __init__(self,
                 config: PretrainedConfig,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 add_bos_token: bool = True,
                 ):
        super().__init__(config)
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=add_bos_token)
        # The default behaviour of GPT2 is not to add bos_token in the beginning of the sequence, but most LLMs
        # have bos token and use it, so we chose to change this default behaviour.
        self.add_bos_token = add_bos_token
        self.context_window_size = GPT2_WINDOW_SIZE
        self._adapt_weights()

    def _adapt_weights(self):
        # We need to override the regular loading of wpe weight since we are adding support to longer contexts.
        self.transformer.wpe = GPT2LMHeadModel.from_pretrained(self.config.name_or_path).transformer.wpe

    def _get_windows(self, texts: List[str]) -> List[Dict]:
        windows = []
        for text in texts:
            encoded_input_window = self.tokenizer(text, return_tensors='pt').to(self.device)
            output = self(**encoded_input_window)
            windows.append({'text': text,
                            'encoded_input': encoded_input_window,
                            'attention_mask': encoded_input_window['attention_mask'],
                            'window_size': encoded_input_window['input_ids'].shape[1],
                            'output': output,
                            'past': output['past_key_values']})
        return windows

    def get_contexts_cache(self, contexts: List[str]) -> Dict:
        windows = self._get_windows(contexts)
        res = {'past_key_values': combine_past_key_values([window['past'] for window in windows],
                                                          contains_bos_token=self.add_bos_token),
               'max_window_size': max(window['window_size'] for window in windows)}

        if self.add_bos_token:  # if windows contain bos tokens, we remove all but one to avoid multiple bos
            res['past_attention_mask'] = torch.cat([windows[0]['attention_mask']] + [window['attention_mask'][:, 1:]
                                                                                     for window in windows[1:]], dim=1)
            res['sum_windows_size'] = sum(window['window_size'] for window in windows) - (len(windows) - 1)
        else:
            res['past_attention_mask'] = torch.cat([window['attention_mask'] for window in windows], dim=1)
            res['sum_windows_size'] = sum(window['window_size'] for window in windows)
        return res

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
        combined_attention_mask = torch.cat((cache['past_attention_mask'], encoded_task_text['attention_mask']), dim=1)
        res = self.generate(input_ids=encoded_task_text['input_ids'],
                            attention_mask=combined_attention_mask,
                            windows_key_values=cache['past_key_values'],
                            max_window_size=cache['max_window_size'],
                            sum_windows_size=cache['sum_windows_size'],
                            pad_token_id=self.tokenizer.eos_token_id,
                            **kwargs)[0]
        res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
        return self.tokenizer.decode(res[encoded_task_text['input_ids'].shape[1]:])

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.LongTensor,
                                      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      windows_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      max_window_size: Optional[int] = None,
                                      sum_windows_size: Optional[int] = None,
                                      **kwargs
                                      ) -> Dict:
        """input_ids:
            ids of task_tokens.
         attention_mask:
            concatenation of windows + task tokens attentions masks.

         Note (past_key_values vs windows_key_values):
             In the first token generation, past_key_values is None while windows_key_values contains the combined past
             key values of context windows. During following generations, past_key_values is the concatenation of
             windows_key_values + previous generations. Thus, windows_key_values is practically ignored.
             """

        token_type_ids = kwargs.get("token_type_ids")
        # only last token for inputs_ids if past_key_values is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")

        if attention_mask is not None and position_ids is None:
            # create PCW's position_ids on the fly
            position_ids = attention_mask.long().cumsum(-1) - 1
            n_task_tokens = position_ids.shape[1] - sum_windows_size
            position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:  # i.e., first token is already generated
                position_ids = position_ids[:, -1].unsqueeze(-1)
            elif windows_key_values:  # i.e., we are in the first token generation
                position_ids = position_ids[:, sum_windows_size:]
        else:
            position_ids = None

        if windows_key_values and not past_key_values:
            past_key_values = windows_key_values
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
