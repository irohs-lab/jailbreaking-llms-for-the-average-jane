import os
import pandas as pd
import numpy as np
import hydra
import json
from tqdm import tqdm
from pprint import pprint
import contextlib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

from config.model import get_model_config_name
from src.jailbreaks import jailbreaks
from src.pipeline.online_learning import get_data, get_expert_data, get_ol_kwargs, learn_weights

JAILBREAK = "octal_encode"
TARGET_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
OL_SCHEME = "thompson_sampling"


def build_cfg(target_model: str, ol_scheme:str):
    with hydra.initialize(config_path="config"):
        cfg = hydra.compose(
            config_name="main",
            overrides=[
                f"+eval/jailbreak_set=full",
                f"eval.target_model.model_name={target_model}",
                f"eval.target_model.backend=vllm",
                f"+eval.target_model.gen_kwargs.max_new_tokens=2048",
                f"paths.expert_data_dir=/scratch/prarabdh/data",
                f"eval.use_ol=True",
                f"+llm_judge=rate",
                f"llm_judge.judge_type=rater",
                f"llm_judge.model.model_name=google/gemma-3-27b-it",
                f"+llm_judge.model.gen_kwargs.max_new_tokens=32",
                f"llm_judge.use_icl=False",
                f"llm_judge.prompt_id=rating_prompt_binary_v1_no_icl",
                f"llm_judge.output_mode=rating_only",
                f"+ol_scheme={ol_scheme}",
                f"attack.continual=False"
            ]
        )
    return cfg

def main():

    cfg = build_cfg(TARGET_MODEL, OL_SCHEME)
    cfg.eval.jailbreaks = cfg.eval.jailbreak_set.jailbreaks

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            data = get_data(cfg)['data']
            output = get_expert_data(cfg, data)

    expert_data = output['expert_data']
    T,n = expert_data['train'].shape[0], expert_data['train'].shape[1]

    jb_list = list(cfg.eval.jailbreaks)
    jb_idx = jb_list.index(JAILBREAK)

    train_rewards = 1 - expert_data['train']
    rows_to_keep = np.where(train_rewards[:, jb_idx] == 1)[0]

    all_idx = np.arange(len(data['train']))
    rows_rest = np.setdiff1d(all_idx, rows_to_keep, assume_unique=True)

    new_order = np.concatenate([rows_to_keep, rows_rest])
    data['train'] = data['train'].iloc[new_order]
    expert_data['train'] = expert_data['train'][new_order]

    kwargs = get_ol_kwargs(cfg, data, expert_data, "test")

    ol_kwargs = {
        k:v
        for k,v in kwargs.items() if k not in ('data', 'expert_data')
    }

    ol_output = learn_weights(OL_SCHEME)(cfg, data, expert_data, **ol_kwargs)

    print(f"{OL_SCHEME} Test ASR: {ol_output['observed_asr']['test']}")

    # Manually run BCBF attack

    sample_size = ol_kwargs['T']//ol_kwargs['n']
    sampled_train_data = data['train'].iloc[:sample_size]
    sampled_expert = expert_data['train'][:sample_size]

    sample_train_rewards = 1 - sampled_expert
    average_asr:np.ndarray = sample_train_rewards.mean(axis=0)

    a = average_asr.argmax()

    test_rewards = 1 - expert_data['test']
    bcbf_test_asr = test_rewards.mean(axis=0)[a]

    print(f"BCBF Chosen Jailbreak: {jb_list[a]}")
    print(f"BCBF Test ASR:{bcbf_test_asr}")

if __name__ == "__main__":
    main()