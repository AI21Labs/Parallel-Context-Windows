import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import typing as npt
from torch import distributed as dist
from transformers import PreTrainedTokenizerBase, LlamaTokenizer

from constants import TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_max_n_shots(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase,
                    prompt_size: int) -> int:
    n_tokens_between_shots = n_tokens_in_prompt(tokenizer, TEXT_BETWEEN_SHOTS)
    shot_lengths = train_df[N_TOKENS] + n_tokens_between_shots
    prompt_length_percentile = shot_lengths.quantile(0.9)
    longest_test_prompt = test_df[N_TOKENS].max()
    _logger.info(f"longest_test_prompt = {longest_test_prompt}")
    max_possible_shots_length = prompt_size - longest_test_prompt
    return int(np.floor(max_possible_shots_length / prompt_length_percentile))


def filter_extremely_long_samples(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    df[N_TOKENS] = df[PROMPTS].map(lambda x: n_tokens_in_prompt(tokenizer, x))
    mask = df[N_TOKENS] <= df[N_TOKENS].quantile(0.99)
    _logger.info(f"filtered {sum(~mask)} from  dataset due to extreme length")
    df = df.loc[mask].copy()
    _logger.info(f"longest remaining prompt according to tokenizer: {df[N_TOKENS].max()}")
    return df


def n_tokens_in_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str, add_special_tokens=False) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def plot_results_graph(results, dataset_name, n_shots, model='') -> None:
    plt.figure()
    plt.errorbar(n_shots, np.mean(results, axis=1), np.std(results, axis=1), fmt='*')
    plt.xlabel("# shots")
    plt.xticks(n_shots)
    metric = 'Accuracy'
    plt.ylabel(f"{dataset_name} {metric}")
    plt.title(f"{metric} {dataset_name} {model}")


def load_results(dataset_name: str, output_dir: str, plot=False) -> Tuple[npt.NDArray[float], List[int]]:
    all_results = os.listdir(output_dir)
    results_path = [r for r in all_results if r.startswith(f'{dataset_name}_')]
    if len(results_path) != 1:
        raise ValueError(f"Found {len(results_path)} results!")
    results_path = results_path[0]
    results = np.load(os.path.join(output_dir, results_path))
    n_shots = [int(d) for d in results_path.split('.')[-2].split('_') if d.isdigit()]
    if plot:
        plot_results_graph(results, dataset_name, n_shots)
    return results, n_shots


def save_results(dataset: str, n_shots: List[int], results: npt.NDArray[int], output_dir: str,
                 model: str = '', plot_results: bool = True) -> None:
    if plot_results:
        plot_results_graph(results, dataset, n_shots, model)
        plt.show()
    if not dist.is_initialized() or dist.get_rank() == 0:
        # in case we use multiple GPUs - we only save one file
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{dataset}_n_shots_results_{'_'.join([str(i) for i in n_shots])}.npy"
        np.save(output_path, results)


def encode_labels(tokenizer: PreTrainedTokenizerBase, labels: List[str]) -> List[List[int]]:
    if isinstance(tokenizer, LlamaTokenizer):
        # sentence piece - adds a space at the beginning of the sentence
        return [tokenizer.encode(f'{label.lstrip()}', add_special_tokens=False) for label in labels]

    return [tokenizer.encode(f' {label.lstrip()}', add_special_tokens=False) for label in labels]


def encode_stop_seq(tokenizer: PreTrainedTokenizerBase, stop_seq: str) -> int:
    stop_seq_token_id = tokenizer.encode(stop_seq, add_special_tokens=False)
    if isinstance(tokenizer, LlamaTokenizer):
        assert len(stop_seq_token_id) == 2
    else:
        assert len(stop_seq_token_id) == 1
    return stop_seq_token_id[-1]
