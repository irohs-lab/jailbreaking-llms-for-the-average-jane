import pandas as pd
import os
import numpy as np
from rich.console import Console
from rich.table import Table
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import re
import json
from functools import partial
from tqdm import tqdm
import fcntl

import config.main
from config.main import get_ol_cfg_name
from config.model import get_model_config_name
from src.utils.inference import run_inference
from src.utils.parser import parse_output
import src.jailbreaks as jailbreaks
from src.utils.tasks import PIPELINE_ABORT, conditional


def setup_cross_model_transfer(cfg:config.main.ProjectConfig):

    if cfg.eval.jailbreak_set is not None:
        cfg.eval.jailbreaks = cfg.eval.jailbreak_set.jailbreaks
    
    console = Console()

    table = Table(show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Target Model", cfg.cmt_config.target_model.model_name)

    table.add_row("Prior Model", cfg.cmt_config.prior_model.model_name)

    table.add_row("Judge Model", cfg.llm_judge.model.model_name)

    table.add_row("Using Local Cache", str(cfg.cmt_config.reuse_local_cache))

    num_jb = len(cfg.eval.jailbreaks)

    hydra_choices = HydraConfig.get().runtime.choices
    jb_set_name = hydra_choices.get("eval/jailbreak_set", None)

    if jb_set_name is not None:
        
        if jailbreaks == [None]:
            jailbreak_str = "baseline (1 jailbreak)"
        else:
            if num_jb == 1:
                jailbreak_str = f"{jb_set_name} ({num_jb} jailbreak)"
            else:
                jailbreak_str = f"{jb_set_name} ({num_jb} jailbreaks)"
    else:

        named = ["baseline" if jb is None else jb for jb in cfg.eval.jailbreaks]

        if num_jb <=3:
            jailbreak_str = f"{', '.join(named)} ({num_jb})"
        else:
            preview = ", ".join(named[:3])
            jailbreak_str = f"{preview}, … ({num_jb} jailbreaks)"
    
    table.add_row("Jailbreaks", jailbreak_str)

    table.add_row("OL Algorithm", cfg.ol_scheme.display_name)
    table.add_row("Num Passes", str(cfg.attack.num_passes))
    
    ol_dict = OmegaConf.to_container(cfg.ol_scheme, resolve=True)

    for k,v in ol_dict.items():
        if k in ('experts', 'name', 'display_name'): continue
        table.add_row(k,str(v))
    
    console.print(table)

def load_test_data(cfg:config.main.ProjectConfig):

    data_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.benchmark_dir
    )

    data = pd.read_csv(
        os.path.join(
            data_root,
            f"test.csv"
        )
    )

    data['split'] = 'test'

    return dict(data=data)

def load_ol_weights(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    """
        Loads transfer attack weights on the test set. For non-contextual algorithms, this is a rank-1 matrix (all rows identical to be specific). 

    """

    prior_model_dir = cfg.cmt_config.prior_model.model_name.split('/')[-1].replace('-','_')
    weight_log_root = os.path.join(
        cfg.paths.result_dir,
        "transfer",
        get_ol_cfg_name(cfg),
        "traindomains_FHLCEPtestdomains_FHLCEP",
        prior_model_dir
    )

    weight_log_files = [f for f in os.listdir(weight_log_root) if (f.endswith(".csv") and "weight_log" in f)]

    max_train_rounds = 0
    max_tr_rounds_files = []

    for wlf in weight_log_files:
        match = re.search(r"trainRounds_(\d+)_", wlf)

        if match:
            train_rounds = int(match.group(1))
            if train_rounds > max_train_rounds: 
                max_train_rounds = train_rounds
    
    expert_set = HydraConfig.get().runtime.choices.get("eval/jailbreak_set", None)
    for wlf in weight_log_files:
        match = re.search(rf"trainRounds_{max_train_rounds}_testRounds_\d+_num_jailbreaks_{len(cfg.eval.jailbreaks)}{f'_expert_set_{expert_set}' if expert_set is not None else ''}_weight_log_test.csv", wlf)

        if match:
            max_tr_rounds_files.append(match.group())
    
    if len(max_tr_rounds_files) > 1:
        print("Found more than 1 train weight log file with max training rounds. Please specify the one you want to use:")
        for idx, file in enumerate(max_tr_rounds_files):
            print(f"[{idx}]. {file}")
        
        idx = input(f"Enter the file index to use (0-{len(max_tr_rounds_files)}):")
        weight_log_file = max_tr_rounds_files[idx]
    else:
        weight_log_file = max_tr_rounds_files[0]
    
    wl_path = os.path.join(
        weight_log_root,
        weight_log_file
    )

    weight_log_df = pd.read_csv(wl_path)

    assert list(weight_log_df.columns) == [jb if jb is not None else 'baseline' for jb in cfg.eval.jailbreaks]

    weight_log = weight_log_df.to_numpy()

    return dict(weight_log=weight_log)

def load_expert_data(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    cache_root = os.path.join(
        cfg.paths.expert_data_dir,
        cfg.paths.cache_dir,
        "target_model_responses"
    )

    target_model = cfg.cmt_config.target_model.model_name.split('/')[-1].replace('-','_')
    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"

    judge_key = (
        get_model_config_name(cfg.llm_judge.model, not_hash=True)
        + f"__{cfg.llm_judge.prompt_id}"
        + f"__{cfg.llm_judge.output_mode}"
        + f"__{icl_tag}"
    )

    target_key = get_model_config_name(cfg.cmt_config.target_model, not_hash=True)

    train_domains = cfg.attack.train_domains
    test_domains = cfg.attack.test_domains

    experts = [exp if exp is not None else 'baseline' for exp in cfg.eval.jailbreaks]

    n = len(experts)

    expert_data = np.zeros((len(data),n))

    total_steps = len(test_domains)*len(experts)

    hash_rating_map = {}

    with tqdm(total=total_steps, desc="Loading expert data") as pbar:
        domain_set = test_domains
        for jb_idx, jb in enumerate(experts):
            for domain in domain_set:
                expert_file = os.path.join(
                    cache_root,
                    domain,
                    jb,
                    target_model,
                    "test", 
                    f"{target_key}__judged_{judge_key}.csv"
                )

                manual_rated_file = os.path.join(
                    cache_root,
                    domain,
                    jb,
                    target_model,
                    "test",
                    f"{target_key}__judged_{judge_key}_manually_rated.csv"
                )

                if os.path.exists(manual_rated_file):
                    expert_df = pd.read_csv(manual_rated_file)
                    rating_col = 'manual_rating'
                else:
                    expert_df = pd.read_csv(expert_file)
                    rating_col = 'judge_rating'
                
                expert_df.loc[
                    expert_df['model_response'].isna(), rating_col
                ] = 1.0

                hash_rating_map.update(**dict(zip(expert_df['prompt_hash'], expert_df[rating_col])))
                pbar.update(1)

            ratings = data["prompt_hash"].map(hash_rating_map)
            expert_data[:,jb_idx] = ratings.to_numpy()

    return dict(expert_data=expert_data)

def run_ol_attack(cfg:config.main.ProjectConfig, data:pd.DataFrame, weight_log:np.ndarray, expert_data:np.ndarray):

    if cfg.ol_scheme.name in ('randomized_weighted_majority', 'exp3'):
        
        weights = weight_log[0,:]
        rng = np.random.default_rng(seed=cfg.seed)
        n = weights.shape[0]
        T = expert_data.shape[0]

        assert n == len(cfg.eval.jailbreaks)
        assert n == expert_data.shape[-1]
        assert T == len(data)

        observed_asr = 0
        
        for t in range(T):
            a_t = rng.choice(n,p=weights)
            r_it = 1-expert_data[t,a_t]
            observed_asr+=r_it
        
        observed_asr/=T
    elif cfg.ol_scheme.name in ('linucb', 'thompson_sampling'):

        weights = weight_log
        n = weights.shape[1]
        T = expert_data.shape[0]

        assert n == len(cfg.eval.jailbreaks)
        assert n == expert_data.shape[-1]
        assert T == len(data)
        assert T == weights.shape[0]

        observed_asr = 0

        for t in range(T):
            p_t = weights[t,:]
            a_t = np.argmax(p_t)
            r_it = 1-expert_data[t,a_t]
            observed_asr+=r_it
        
        observed_asr/=T
    elif cfg.ol_scheme.name in ('linear_cb', 'square_cb'):

        weights = weight_log
        rng = np.random.default_rng(seed=cfg.seed)
        n = weights.shape[1]
        T = expert_data.shape[0]

        assert n == len(cfg.eval.jailbreaks)
        assert n == expert_data.shape[-1]
        assert T == len(data)
        assert T == weights.shape[0]

        observed_asr = 0

        for t in range(T):
            p_t = weights[t,:]
            a_t = rng.choice(n,p=p_t)
            r_it = 1 - expert_data[t,a_t]

            observed_asr+=r_it
        
        observed_asr/=T
    else:
        raise ValueError(f"OL Scheme {cfg.ol_scheme.name} not supported")
        

    return dict(observed_asr=observed_asr, num_examples=T, num_sanitized=None)
    

def construct_inference_dataframe(cfg:config.main.ProjectConfig, data:pd.DataFrame, weight_log:np.ndarray):

    if cfg.ol_scheme.name in ('linear_cb', 'linucb', 'square_cb'):
        raise NotImplementedError(f"Contextual OL Algorithms not supported yet for transferring the attack to proprietary models. Use local cache if you're transferring to an open-weight model.")
    else:
        assert (weight_log[0,:] == weight_log[-32,:]).all(), "weight log does not have identical rows!" # checking if the weight log has identical rows based on a comparison b/w the first row and a randomly chosen row
        weights = weight_log[0,:]

    cache_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "jailbreak_prompts"
    )

    os.makedirs(cache_root, exist_ok=True)

    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}.csv"

    cache_path = os.path.join(cache_root, cache_file)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return dict(to_infer_df=df)


    rng = np.random.default_rng(seed=cfg.seed)
    n = weights.shape[0]

    assert n == len(cfg.eval.jailbreaks)
    jb_prompts = []
    jb_names = []

    for idx, row in data.iterrows():
        a_t = rng.choice(n,p=weights)
        jb_name = cfg.eval.jailbreaks[a_t]
        if jb_name is not None:
            jb_func = getattr(jailbreaks, jb_name)
        else:
            jb_func = lambda x:x
            jb_name = 'baseline'
        
        jb_prompts.append(jb_func(row['prompt']))
        jb_names.append(jb_name)
    
    data['jb_prompt'] = jb_prompts
    data['jailbreak'] = jb_names
    
    data.to_csv(cache_path, index=False)
       
    return dict(to_infer_df=data)

def check_cache_for_target_model_responses(cfg:config.main.ProjectConfig):

    save_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "target_model_responses"
    )

    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}.csv"

    cache_path = os.path.join(save_root, cache_file)

    return os.path.exists(cache_path)

@conditional(lambda to_infer_df: to_infer_df is not None and len(to_infer_df) > 0)
def run_target_inference(cfg:config.main.ProjectConfig, to_infer_df:pd.DataFrame):

    model = cfg.cmt_config.target_model.model_name
    backend = cfg.cmt_config.target_model.backend
    gen_kwargs = cfg.cmt_config.target_model.gen_kwargs

    responses = run_inference(backend, model, to_infer_df['jb_prompt'].tolist(), dict(cfg.cmt_config.target_model.engine_kwargs), **gen_kwargs)

    if responses is not None:
        to_infer_df['model_response'] = responses

        return dict(to_infer_df=to_infer_df)

    else: 
        if backend in ('openai', 'claude', 'grok'):
            return PIPELINE_ABORT
        else:
            raise ValueError(f"run_inference() returned None value")

def save_target_model_responses(cfg:config.main.ProjectConfig, to_infer_df:pd.DataFrame):

    save_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "target_model_responses"
    )

    os.makedirs(save_root, exist_ok=True)
    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}.csv"

    cache_path = os.path.join(save_root, cache_file)

    to_infer_df.to_csv(cache_path, index=False)


def load_cached_target_model_responses(cfg:config.main.ProjectConfig):
    
    save_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "target_model_responses"
    )

    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}.csv"

    cache_path = os.path.join(save_root, cache_file)

    inferred_df = pd.read_csv(cache_path)

    return dict(inferred_df=inferred_df)

def check_judge_cache(cfg:config.main.ProjectConfig):
    save_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "target_model_responses"
    )

    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}__judged.csv"

    cache_path = os.path.join(save_root, cache_file)

    return os.path.exists(cache_path)

def automatically_rate_failed_responses(cfg:config.main.ProjectConfig, inferred_df:pd.DataFrame):

    # If the model did not respond, or the prompt was caught by input sanitation, automatically rate it as '1' (i.e., failure)
    inferred_df.loc[
        inferred_df['model_response'] == 'ERR', 'judge_rating'
    ] = 1

    inferred_df.loc[
        inferred_df['model_response'].isna(), 'judge_rating'
    ] = 1

    return dict(inferred_df=inferred_df)

@conditional(lambda inferred_df: inferred_df is not None)
def prepare_judge_inputs(cfg:config.main.ProjectConfig, inferred_df:pd.DataFrame):

    to_judge_df = inferred_df.loc[inferred_df['judge_rating'].isna(),:]
  
    icl_file = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.icl_file
    )
    
    example_strings = {}

    if cfg.llm_judge.use_icl:
        with open(icl_file, "r") as f:
            icl_examples = json.load(f)[cfg.llm_judge.icl_id]

        for domain in icl_examples.keys():
            example_strings[domain] = ""
            for ex in icl_examples[domain]:
                if cfg.llm_judge.label_space == [0,1] and ex['rating'] == 0.5:
                    continue
                if cfg.llm_judge.output_mode == "rating_only":
                    example_strings[domain] += (
                        f"Query: {ex['query']}\n"
                        f"Response: {ex['response']}\n"
                        f"Rating: {ex['rating']}\n\n"
                    )
                else:
                    example_strings[domain] += (
                        f"Query: {ex['query']}\n"
                        f"Response: {ex['response']}\n"
                        f"Rating: {ex['rating']}\n"
                        f"Reason: {ex['reason']}\n\n"
                    )
    else:
        for domain in to_judge_df["domain"].unique():
            example_strings[domain] = ""
    
    
    with open(os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.prompts_file
    ), "r") as f:
        judge_prompt = json.load(f)[cfg.llm_judge.prompt_id]['text']
    
    domain_wise_judge_prompts = {}

    for domain in example_strings.keys():
        if cfg.llm_judge.use_icl:
            domain_wise_judge_prompts[domain] = judge_prompt.format(domain=domain, icl_examples=example_strings[domain])
        else:
            domain_wise_judge_prompts[domain] = judge_prompt.format(domain=domain)
    
    to_judge_df["judge_prompt_template"] = to_judge_df["domain"].map(
    domain_wise_judge_prompts
    )

    if to_judge_df["judge_prompt_template"].isna().any():
        missing = to_judge_df.loc[
            to_judge_df["judge_prompt_template"].isna(), "domain"
        ].unique()
        raise ValueError(f"Missing judge prompt templates for domains: {missing}")
    
    to_judge_df["judge_prompt"] = (
        to_judge_df["judge_prompt_template"]
        + "Query: "
        + to_judge_df["prompt"]
        + "\nResponse:"
        + to_judge_df["model_response"].fillna("")
    )

    to_judge_df.drop(columns="judge_prompt_template", inplace=True)
    
    return dict(to_judge_df=to_judge_df, inferred_df=inferred_df)

@conditional(lambda to_judge_df:to_judge_df is not None)
def get_ratings_from_judge(cfg:config.main.ProjectConfig, to_judge_df:pd.DataFrame, inferred_df:pd.DataFrame):

    model = cfg.llm_judge.model.model_name
    backend = cfg.llm_judge.model.backend
    gen_kwargs = cfg.llm_judge.model.gen_kwargs

    inputs = to_judge_df['judge_prompt'].tolist()

    responses = run_inference(backend, model, inputs, dict(cfg.llm_judge.model.engine_kwargs), **gen_kwargs)

    if responses is not None:
        to_judge_df['judge_response'] = responses

        return dict(judged_df=to_judge_df, inferred_df=inferred_df)
    else:
        if backend in ('openai', 'claude'):
            return PIPELINE_ABORT
        else:
            raise ValueError("run_inference returned None")
    
@conditional(lambda judged_df: judged_df is not None)
def parse_judge_outputs(cfg:config.main.ProjectConfig, judged_df:pd.DataFrame, inferred_df:pd.DataFrame):

    parse_func = partial(parse_output, parse_type=cfg.llm_judge.judge_type, output_mode=cfg.llm_judge.output_mode)

    judged_df['judge_rating'] = judged_df['judge_response'].apply(parse_func)

    judged_df['judge_rating'] = pd.to_numeric(judged_df['judge_rating'], errors='coerce')

    missing_rows = inferred_df.loc[~inferred_df['prompt_hash'].isin(judged_df['prompt_hash'])]

    judged_df = pd.concat([judged_df, missing_rows], ignore_index=True)

    assert len(judged_df) == len(inferred_df)

    return dict(judged_df=judged_df)


def save_judged_outputs(cfg:config.main.ProjectConfig, judged_df:pd.DataFrame):

    save_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "target_model_responses"
    )

    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}__judged.csv"

    cache_path = os.path.join(save_root, cache_file)

    judged_df.to_csv(cache_path, index=False)

    return dict(judged_df=judged_df)

def load_judge_cache(cfg:config.main.ProjectConfig):
    save_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cmt_cache_dir,
        "target_model_responses"
    )

    cache_file = f"target_model_{get_model_config_name(cfg.cmt_config.target_model)}_prior_model_{get_model_config_name(cfg.cmt_config.prior_model)}_judge_model_{get_model_config_name(cfg.llm_judge.model)}_{get_ol_cfg_name(cfg)}__judged.csv"

    cache_path = os.path.join(save_root, cache_file)

    df = pd.read_csv(cache_path)
    
    return dict(judged_df=df)

def compute_metrics(cfg:config.main.ProjectConfig, judged_df:pd.DataFrame):

    ratings = judged_df['judge_rating'].to_numpy()
    successes = 1-ratings
    observed_asr = successes.mean()

    num_examples = len(judged_df)
    num_sanitized = judged_df['model_response'].value_counts().to_dict()['ERR']

    return dict(observed_asr=observed_asr, num_examples=num_examples, num_sanitized=num_sanitized)
    
def display_summary(cfg:config.main.ProjectConfig, observed_asr, num_examples, num_sanitized=None):
    console = Console()
    table = Table(show_header=False)
    table.add_column("Field", style="bold magenta")
    table.add_column("Value", style="bold turquoise4")
    
    table.add_row("Num Examples", str(num_examples))
    table.add_row("Attack Success Rate", f"{observed_asr*100:0.2f}")
    if num_sanitized:
        table.add_row("Num Sanitized", str(num_sanitized))

    console.print(table)

def save_results(cfg, observed_asr):
    save_file = os.path.join(
        cfg.paths.result_dir,
        "open_weight_cmt_results.json"
    )

    os.makedirs(cfg.paths.result_dir, exist_ok=True)

    with open(save_file, "a+") as f:

        fcntl.flock(f, fcntl.LOCK_EX)

        f.seek(0)

        try:
            results = json.load(f)
        except json.JSONDecodeError:
            results = {}

        scheme = cfg.ol_scheme.name
        target = cfg.cmt_config.target_model.model_name
        prior = cfg.cmt_config.prior_model.model_name

        results.setdefault(scheme, {})
        results[scheme].setdefault(target, {})

        if prior in results[scheme][target] and results[scheme][target][prior] != observed_asr:
            print(f"\nResult already exists for:")
            print(f"  Scheme: {scheme}")
            print(f"  Target: {target}")
            print(f"  Prior : {prior}")
            print(f"  Existing value: {results[scheme][target][prior]}")
            print(f"  Current value: {observed_asr}")
            print("Skipping overwrite in parallel mode.")
            fcntl.flock(f, fcntl.LOCK_UN)
            return False

        results[scheme][target][prior] = observed_asr

        f.seek(0)
        f.truncate()
        json.dump(results, f, indent=4)

        fcntl.flock(f, fcntl.LOCK_UN)

    return True

def get_tasks(cfg:config.main.ProjectConfig):

    if cfg.cmt_config.reuse_local_cache:
        tasks = [
            setup_cross_model_transfer,
            load_test_data,
            load_ol_weights,
            load_expert_data,
            run_ol_attack,
            display_summary,
            save_results

        ]
    else:
        tasks = [
            setup_cross_model_transfer,
            load_test_data,
            load_ol_weights,
            construct_inference_dataframe,
        ]

        if not check_cache_for_target_model_responses(cfg):
            tasks.extend([
                run_target_inference,
                save_target_model_responses
            ])
        else:
            tasks.extend([
                load_cached_target_model_responses
            ])
        
        if not check_judge_cache(cfg):
            tasks.extend([
                automatically_rate_failed_responses,
                prepare_judge_inputs,
                get_ratings_from_judge,
                parse_judge_outputs,
                save_judged_outputs,
                compute_metrics,
                display_summary
            ])
        else:
            tasks.extend([
                load_judge_cache, 
                compute_metrics,
                display_summary
            ])

    return tasks