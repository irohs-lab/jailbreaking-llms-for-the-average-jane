import os
import json
import copy
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np

import config.main
from config.model import get_model_config_name
from src.pipeline.online_learning.hyperparameters import get_ol_kwargs
from src.pipeline.online_learning.algorithms import learn_weights

EXPERT_CACHE = {}

def get_target_model_key(target_model_name:str, target_model_params:dict):
    name = target_model_name.split('/')[-1].replace('-','_') + "_" + target_model_params['backend']

    for k,v in target_model_params.items():
        if k!='backend':
            name+=f"_{k}_{v}"
    
    return name


def get_expert_data(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame],target_model_name:str,target_model_params:dict):

    data = {k: v.copy() for k, v in data.items()}

    cache_root = os.path.join(
        cfg.paths.expert_data_dir,
        cfg.paths.cache_dir,
        "target_model_responses"
    )

    target_model = target_model_name.split('/')[-1].replace('-','_')
    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"
    judge_key = (
        get_model_config_name(cfg.llm_judge.model, not_hash=True)
        + f"__{cfg.llm_judge.prompt_id}"
        + f"__{cfg.llm_judge.output_mode}"
        + f"__{icl_tag}"
    )
    target_key = get_target_model_key(target_model_name, target_model_params)
    train_domains = cfg.attack.train_domains
    test_domains = cfg.attack.test_domains
    
    expert_data = {}
    experts = [exp if exp is not None else 'baseline' for exp in cfg.eval.jailbreaks]

    n = len(experts)

    for split, split_df in data.items():
        expert_data[split] = np.zeros((len(split_df), n))
        if split == 'train' or split == 'val': 
            domain_set = train_domains
        else:
            domain_set = test_domains
        for jb_idx,jb in enumerate(experts):
            hash_rating_map = {}
            for domain in domain_set:
                expert_file = os.path.join(
                    cache_root,
                    domain,
                    jb,
                    target_model,
                    split,
                    f"{target_key}__judged_{judge_key}.csv"
                )
                manual_rated_file = os.path.join(
                    cache_root,
                    domain,
                    jb,
                    target_model,
                    split,
                    f"{target_key}__judged_{judge_key}_manually_rated.csv"
                )
                if os.path.exists(manual_rated_file):
                    expert_df = pd.read_csv(manual_rated_file)
                    rating_col = 'manual_rating'
                else:
                    expert_df = pd.read_csv(expert_file)
                    rating_col='judge_rating'
                # if expert_df['judge_rating'].isna().any():breakpoint()
                hash_rating_map.update(**dict(zip(expert_df['prompt_hash'], expert_df[rating_col])))
        
            expert_data[split][:, jb_idx] = split_df['prompt_hash'].map(hash_rating_map).to_numpy()

    # for split in expert_data:
    #     mask = ~np.isnan(expert_data[split]).any(axis=1)
    #     expert_data[split] = expert_data[split][mask]
    #     data[split] = data[split].iloc[mask].reset_index(drop=True)

    # for split, split_df in data.items():
    #     print(f"Split: {split}, Split Size: {len(split_df)}")

    return dict(data=data, expert_data=expert_data)

def evaluate_combo(args):

    cfg, data, combo, cache_file = args

    params = combo["params"]
    target_model = combo["target_model"]

    # cfg_copy = copy.deepcopy(cfg)
    cfg_copy = cfg

    for k, v in params.items():
        setattr(cfg_copy.ol_scheme, k, v)

    cfg_copy.eval.target_model.model_name = target_model

    cache_key = (
        target_model,
        tuple(cfg.eval.jailbreaks),
        tuple(cfg.attack.train_domains),
        tuple(cfg.attack.test_domains),
    )

    if cache_key not in EXPERT_CACHE:
        EXPERT_CACHE[cache_key] = get_expert_data(
            cfg_copy,
            data,
            target_model,
            cfg.ol_val_sweep.target_model_params
        )

    expert_out = EXPERT_CACHE[cache_key]

    expert_model = expert_out["expert_data"]
    modified_data = expert_out["data"]

    data_eval = {
        "train": modified_data["train"],
        "val": modified_data["val"]
    }

    expert_data = {
        "train": expert_model["train"],
        "val": expert_model["val"]
    }

    kwargs = get_ol_kwargs(
        cfg_copy,
        data_eval,
        expert_data,
        eval_split="val"
    )

    alg_fn = learn_weights(cfg_copy.ol_scheme.name)

    result = alg_fn(cfg=cfg_copy, **kwargs)

    score = result["observed_asr"]["val"]

    output = dict(
        params=params,
        target_model=target_model,
        score=score
    )

    with open(cache_file, "w") as f:
        json.dump(output, f)

    return output

def run_sweep_parallel(cfg, data, param_grid):

    cache_dir = os.path.join(
        cfg.paths.result_dir,
        cfg.ol_val_sweep.cache_dir
    )

    os.makedirs(cache_dir, exist_ok=True)

    cached_results = []
    jobs = []

    for combo in param_grid:

        cache_file = os.path.join(cache_dir, combo["id"] + ".json")

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached_results.append(json.load(f))
        else:
            jobs.append((cfg, data, combo, cache_file))

    print(f"Loaded {len(cached_results)} cached results")
    print(f"Running {len(jobs)} new jobs")

    results = list(cached_results)

    if jobs:

        with mp.Pool(cfg.ol_val_sweep.num_workers) as pool:

            new_results = list(
                tqdm(
                    pool.imap_unordered(evaluate_combo, jobs),
                    total=len(jobs),
                    desc="Hyperparameter Sweep"
                )
            )

        results.extend(new_results)

    return dict(sweep_results=results)