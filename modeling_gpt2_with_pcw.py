from abc import ABC
from typing import Tuple, Optional, Dict

import torch
from transformers import GPT2LMHeadModel
from transformers.configuration_utils import PretrainedConfig

from pcw_wrapper import generate_pcw_position_ids


class GPT2LMHeadPCW(GPT2LMHeadModel, ABC):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self._adapt_weights()

    def _adapt_weights(self):
        # We need to override the regular loading of wpe weight since we are adding support to longer contexts.
        self.transformer.wpe = GPT2LMHeadModel.from_pretrained(self.config.name_or_path).transformer.wpe

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
            position_ids = generate_pcw_position_ids(attention_mask, max_window_size, past_key_values,
                                                     sum_windows_size, windows_key_values)
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
