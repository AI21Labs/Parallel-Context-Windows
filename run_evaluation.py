import argparse
import logging
from typing import List, Optional

import pandas as pd
import torch
from transformers import PreTrainedTokenizer, AutoConfig

from datasets_loader import DATASET_NAMES2LOADERS
from experiment_manager import ExperimentManager
from modeling_gpt2_with_pcw import GPT2LMHeadWithPCWModel, GPT2_WINDOW_SIZE
from utils import get_max_n_shots, filter_extremely_long_samples, save_results

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_inference_wrapper_and_tokenizer(model_name: str, n_windows: int, add_bos_token: bool) -> GPT2LMHeadWithPCWModel:
    # we override n_positions to bi pass the model's context window size restriction (for gpt2, n_positions determines
    # the causal attention mask matrix dimension). The correct position embeddings (i.e., gpt2's 1024 trained
    # position embeddings) are re-inserted to the model in GPT2LMHeadWithPCWModel initialization.
    config = AutoConfig.from_pretrained(model_name, n_positions=GPT2_WINDOW_SIZE * n_windows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadWithPCWModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True,
                                                   add_bos_token=add_bos_token)
    model.to(device)
    return model


def get_dataset(dataset: str, tokenizer: PreTrainedTokenizer) -> (pd.DataFrame, pd.DataFrame, List):
    da = DATASET_NAMES2LOADERS[dataset]()
    # Filter extremely long samples from both train and test samples:
    _logger.info("filtering test set:")
    test_df = filter_extremely_long_samples(da.test_df, tokenizer)
    _logger.info("filtering train set:")
    train_df = filter_extremely_long_samples(da.train_df, tokenizer)
    return test_df, train_df, da.labels


def run_pcw_experiment(dataset: str, model: str, subsample_test_set: int, output_dir: str,
                       n_windows: List[int], n_shots_per_window: Optional[int], n_runs: int,
                       random_seed: int, add_bos_token: bool) -> None:
    pcw_model = get_inference_wrapper_and_tokenizer(model, max(n_windows), add_bos_token)

    test_df, train_df, labels = get_dataset(dataset, pcw_model.tokenizer)

    if n_shots_per_window is None:
        # default behaviour: we take the maximum number of samples per window
        n_shots_per_window = get_max_n_shots(train_df, test_df, pcw_model.tokenizer, pcw_model.context_window_size)
        _logger.info(f"Found max n shot per window = {n_shots_per_window}")

    n_shots = [i * n_shots_per_window for i in n_windows]

    em = ExperimentManager(test_df, train_df, pcw_model, labels, random_seed=random_seed,
                           n_shots_per_window=n_shots_per_window, subsample_test_set=subsample_test_set)

    accuracies = em.run_experiment_across_shots(n_shots, n_runs)
    save_results(dataset, n_shots, accuracies, output_dir, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', action='store', required=True,
                        help=f'Name of dataset (for example sst2).'
                             f' The supported datasets are: {DATASET_NAMES2LOADERS.keys()}')
    parser.add_argument('--model', dest='model', action='store', default='gpt2',
                        help='HF model name to use, one of: [gpt2,gpt2-medium,gpt2-large,gpt2-xl]')
    parser.add_argument('--subsample-test-set', dest='subsample_test_set', action='store', required=False, type=int,
                        help='Size of test set to use to speed up eval. None means using all test set.')
    parser.add_argument('--output-dir', dest='output_dir', required=False, help="Directory for saving the results",
                        default='./temp', action='store', type=str)
    parser.add_argument('--random-seed', dest='random_seed', required=False, default=42, action='store', type=int)
    parser.add_argument('--n-runs', dest='n_runs',
                        help="Number of times experiments are repeated for every number of windows", action='store',
                        type=int, default=1)
    parser.add_argument('-n', '--n-windows', dest='n_windows', help="Number of parallel context windows",
                        action='append', type=int)
    parser.add_argument('--n-shots-per-window', dest='n_shots_per_window',
                        help="number of examples to fit in each window", type=int, default=None)
    parser.add_argument('--add-bos-token', dest='add_bos_token',
                        help="Add a single shared bos token for all of the windows", action='store_true', default=True)
    parser.add_argument('--no-add-bos-token', dest='add_bos_token',
                        help="Don't use any bos tokens", action='store_false')
    args = parser.parse_args()
    run_pcw_experiment(**vars(args))
