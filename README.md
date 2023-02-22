# Parallel Context Windows (PCW)

This repo contains the code for reproducing the classification experiments on GPT2 models from [AI21 Labs](https://www.ai21.com/)' paper [Parallel Context Windows Improve In-Context Learning of Large Language Models
](https://arxiv.org/abs/2212.10947).  
The code was tested with python 3.10, with CPU and single GPU.

## Setup

To install the required libraries in our repo, run:
```bash
pip install -r requirements.txt
```
To have a Pytorch version specific to your CUDA, [install](https://pytorch.org/) your version before running the above command.

## Evaluation
Due to the fact that the paper's results were based on an earlier implementation of PCW and not [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), the results produced using this code may differ slightly from those shown in the paper.
To reproduce similiar results shown in the appendix for GPT2-XL for a specific dataset (for example SST2), simply run:
```bash
python run_evaluation.py \
--dataset sst2 \
--model gpt2-xl \
--n-windows 1 \
--n-windows 3 \
--subsample-test-set 250 \
--n-runs 30 \
--output-dir $OUTPUT_DIR
```
In this run, PCW's performance is evaluated on a subsample (250 samples) of the full test set. 
The experiment is repeated 30 times (with different random samples of training examples) for each number of windows (in this case - one and three). 
As a default, the script uses as many examples per window as possible. 
Note that using a single window is equivalent to regular ICL settings. Thus, this run should give similar results to those shown in Table 5 for SST2 with GPT2-XL.

The evaluation output is a numpy file (shaped `[2,30]`) found in `$OUTPUT_DIR` with the mean accuracy for each repetition and number of windows.
You could read the file directly with np.load, or use utils.py function to load and plot the results.
See --help for further instructions.

## PCW Usage examples
In the evaluation code, only classification tasks are performed.
The code snippet below shows how PCW can be used both for classification and generation:
```python
from modeling_gpt2_with_pcw import RestrictiveTokensLogitsProcessor, GPT2LMHeadWithPCWModel
import numpy as np

model = GPT2LMHeadWithPCWModel.from_pretrained('gpt2-large')
# use PCW with few shot for classification example:
labels = ['positive', 'negative']
labels_input_ids = np.array(
    [model.tokenizer.encode(f' {label.lstrip()}', add_special_tokens=False) for label in labels])
# using RestrictiveTokensLogitsProcessor forces the output to be one of the labels:
logit_processor = RestrictiveTokensLogitsProcessor(labels_input_ids, eos_token_id=model.tokenizer.eos_token_id)
output = model.pcw_generate(contexts=["Review: Great movie! Sentiment: positive\n",
                                      "Review: Horrible film Sentiment: negative\n"],
                            task_text="Review: I liked it Sentiment:",
                            restrictive_logit_preprocessor=logit_processor,
                            temperature=0,
                            max_new_tokens=1)
print(output.strip())
# use PCW for generation:
output = model.pcw_generate(contexts=["Review: Great movie!\n", "Review: Horrible film\n"],
                            task_text="Review:",
                            temperature=1,
                            do_sample=True,
                            max_new_tokens=16)
print(output)
```

## Citation

If you find our paper or code helpful, please consider citing our paper:
```
@article{ratner2022parallel,
  title={Parallel Context Windows Improve In-Context Learning of Large Language Models},
  author={Ratner, Nir and Levine, Yoav and Belinkov, Yonatan and Ram, Ori and Magar,Inbal and Abend, Omri and Karpas, Ehud and Shashua, Amnon and Leyton-Brown, Kevin and Shoham, Yoav},
  journal={arXiv preprint arXiv:2212.10947},
  year={2022}
}
```