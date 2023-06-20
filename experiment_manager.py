import logging
import random
from typing import List, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from constants import TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS
from datasets_loader import LABEL_TOKENS
from pcw_wrapper import PCWModelWrapper
from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt, encode_labels, encode_stop_seq

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

STOP_SEQUENCE = '\n'


class ExperimentManager:
    def __init__(self, test_df: pd.DataFrame, train_df: pd.DataFrame, model: PCWModelWrapper,
                 labels: List[str] = None, random_seed: int = 42, subsample_test_set: int = 250,
                 n_shots_per_window: int = None):
        if subsample_test_set < len(test_df):
            np.random.seed(random_seed)
            test_df = test_df.sample(subsample_test_set)
        self.test_df = test_df
        self.train_df = train_df
        self.model = model
        self.base_random_seed = random_seed
        self.n_shots_per_window = n_shots_per_window
        self.tokenizer = model.tokenizer
        self._initialize_labels_and_logit_processor(labels)

    def _initialize_labels_and_logit_processor(self, labels: List[str]) -> None:
        _logger.info(f"Provided labels: {labels}")
        labels_tokens = encode_labels(self.tokenizer, labels)
        labels_tokens_array = self.minimize_labels_tokens(labels_tokens)
        _logger.info(f"Provided labels average n_tokens: {np.round(np.mean([len(lt) for lt in labels_tokens]), 3)}")
        # we fix the labels accordingly in the test set:
        shorten_label_tokens = [t[t != self.tokenizer.eos_token_id].tolist() for t in labels_tokens_array]
        _logger.info(
            f"shortened labels average n_tokens: {np.round(np.mean([len(lt) for lt in shorten_label_tokens]), 3)}")
        # Moving the test set label tokens to their shorter version:
        map_labels = {old_label: self.tokenizer.decode(t).lstrip() for old_label, t in
                      zip(labels, shorten_label_tokens)}
        self.test_df[LABEL_TOKENS] = self.test_df[LABEL_TOKENS].map(map_labels)
        pad = len(max(shorten_label_tokens, key=len))
        labels_tokens_array = np.array(
            [i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in shorten_label_tokens])
        self.max_n_tokens = pad
        labels_tokens_array = self.pad_contained_labels_with_stop_seq(shorten_label_tokens, labels_tokens_array)
        self.logit_processor = RestrictiveTokensLogitsProcessor(restrictive_token_ids=labels_tokens_array,
                                                                eos_token_id=self.tokenizer.eos_token_id)
        self.possible_labels = set(map_labels.values())

    def minimize_labels_tokens(self, labels_tokens: List[List[int]]) -> npt.NDArray[int]:
        """
         Minimize the number of tokens per label to be the shortest possible unique one.
        """
        pad = len(max(labels_tokens, key=len))
        labels_tokens_array = np.array([i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in labels_tokens])
        for i, tokens in enumerate(labels_tokens):
            for j in range(len(tokens)):
                labels_with_shared_beginnings = np.sum(
                    np.all(labels_tokens_array[:, :j] == np.array(tokens[:j]), axis=1))
                if labels_with_shared_beginnings == 1:
                    labels_tokens_array[i, j:] = self.tokenizer.eos_token_id
                    break
        return labels_tokens_array

    def pad_contained_labels_with_stop_seq(self, labels_tokens: List, labels_tokens_array: npt.NDArray[int]) \
            -> npt.NDArray[int]:
        """
        In case we have two labels, where one label contains the other label (for example: "A" and "A B") we need
        to allow the restrictive decoding to produce the output "A". We support it by adding "\n" to the shorter label.
        """
        stop_seq_token_id = encode_stop_seq(self.tokenizer, STOP_SEQUENCE)
        for i, tokens in enumerate(labels_tokens):
            labels_with_shared_beginnings = np.sum(
                np.all(labels_tokens_array[:, :len(tokens)] == np.array(tokens), axis=1))
            if labels_with_shared_beginnings > 1:
                _logger.info(f"label{self.tokenizer.decode(tokens)} is the beginning of one of the other labels,"
                             f"adding stop sequence to its end")
                labels_tokens_array[i, len(tokens)] = stop_seq_token_id
        return labels_tokens_array

    def _set_random_seed(self, random_seed: int) -> None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    def get_few_shots_acc(self, windows_few_shot: List[str]) -> float:
        predicted_labels = self.get_predicted_labels(windows_few_shot)
        return self.calc_acc(predicted_labels)

    def get_predicted_labels(self, windows_few_shots: List[str]) -> List[str]:
        windows_cache = self.model.get_contexts_cache(windows_few_shots)
        predicted_labels = []
        for q in self.test_df[PROMPTS]:
            predicted_label = self.predict_label(TEXT_BETWEEN_SHOTS + q, windows_cache)
            predicted_labels.append(predicted_label)
        assert set(predicted_labels).issubset(self.possible_labels)
        return predicted_labels

    def predict_label(self, task_text: str, cache: Dict) -> str:
        assert task_text == task_text.rstrip(), "prompt ends with a space!"
        res = self.model.pcw_generate(task_text=task_text,
                                      contexts_cache=cache,
                                      restrictive_logit_preprocessor=self.logit_processor,
                                      temperature=0,
                                      max_new_tokens=self.max_n_tokens)

        return res.lstrip().strip(STOP_SEQUENCE)

    def calc_acc(self, predicted_labels: List) -> float:
        predicted_labels = pd.Series(predicted_labels, index=self.test_df.index)
        acc = np.mean(predicted_labels == self.test_df[LABEL_TOKENS])
        _logger.info(f"accuracy = {np.round(acc, 3)}")
        return acc

    def run_experiment_across_shots(self, n_shots_to_test: List[int], n_runs: int,
                                    too_long_patience: float = 0.2):
        accuracies = np.zeros((len(n_shots_to_test), n_runs))
        for i, n_shots in enumerate(tqdm(n_shots_to_test)):
            _logger.info(f"starting with n = {n_shots}")
            self._set_random_seed(self.base_random_seed + n_shots)
            j = 0
            n_errors = 0
            while j < n_runs:
                few_shots_idx = self.sample_n_shots(n_shots)
                few_shots_prompts = list(self.train_df.loc[few_shots_idx, PROMPTS])
                windows_few_shots = self.build_windows_few_shots_text(few_shots_prompts, self.n_shots_per_window)
                longest_window_n_tokens = max(n_tokens_in_prompt(self.tokenizer, window)
                                              for window in windows_few_shots)
                n_tokens_between_shots = n_tokens_in_prompt(self.tokenizer, TEXT_BETWEEN_SHOTS)
                if (longest_window_n_tokens + n_tokens_between_shots + self.test_df[N_TOKENS].max()
                        + self.max_n_tokens) > self.model.context_window_size:
                    _logger.warning("Drawn training shots were too long, trying again")
                    n_errors += 1
                    assert n_errors <= too_long_patience * n_runs, "too many long inputs were drawn!"
                    continue
                accuracies[i, j] = self.get_few_shots_acc(windows_few_shots)
                j += 1
        return accuracies

    def sample_n_shots(self, n_shots: int) -> npt.NDArray[int]:
        few_shots_df = self.train_df.sample(n_shots)
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        window_size = self.n_shots_per_window or n_shots
        n_windows = int(len(few_shots_df) / window_size)
        if not self.n_shots_per_window or n_windows == 1:
            return few_shots_df.index
        return self.balance_windows_sizes(n_windows, few_shots_df)

    def balance_windows_sizes(self, n_windows: int, few_shots_df: pd.DataFrame) -> npt.NDArray[int]:
        few_shots_df.sort_values(by=N_TOKENS, inplace=True, ascending=False)
        shape = (self.n_shots_per_window, n_windows)
        indexes = np.array(few_shots_df.index).reshape(shape)
        sizes = few_shots_df.loc[indexes.flatten()].n_tokens.values.reshape(indexes.shape)
        for i in range(1, self.n_shots_per_window):
            order = np.argsort((np.sum(sizes[:i, :], axis=0)))
            sizes[i, :] = sizes[i, order]
            indexes[i, :] = indexes[i, order]
        # shuffle the order in each window:
        for i in range(n_windows):
            np.random.shuffle(indexes[:, i])
        indexes = indexes.T.flatten()
        return indexes

    @staticmethod
    def build_windows_few_shots_text(few_shots_prompts: List, window_size: int) -> List[str]:
        if window_size is None:
            window_size = len(few_shots_prompts)
        return [TEXT_BETWEEN_SHOTS.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]
