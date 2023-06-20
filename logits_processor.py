import numpy as np
import torch
from numpy import typing as npt
from transformers import LogitsProcessor

LOGIT_BIAS = 100


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
                (self.restrictive_token_ids, np.ones(self.restrictive_token_ids.shape[0]) * self.eos_token_id)). \
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
