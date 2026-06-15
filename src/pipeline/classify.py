import pandas as pd
import os
import json
from functools import partial
from hydra.core.hydra_config import HydraConfig
import textstat

import config.main
from config.model import get_model_config_name
from src.utils.inference import run_inference
from src.utils.tasks import conditional, PIPELINE_ABORT
from src.utils.parser import parse_output
from src.pipeline.validation.inference import format_icl_examples
from src.pipeline.validation.metrics import get_label_from_readability_score, aggregate

def warn_and_confirm(message):
    print(f"WARNING: {message}")
    response = input("Do you want to continue? [y/N]: ").strip().lower()
    if response not in ("y", "yes"):
        raise RuntimeError("Operation cancelled by user.")

def load_data(cfg:config.main.ProjectConfig):
    splits = ('train', 'test', 'val')

    data_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.benchmark_dir
    )

    split_dfs = []
    for split in splits:
        split_file = os.path.join(data_dir, f"{split}.csv")
        split_df = pd.read_csv(split_file)
        split_df['split'] = split
        split_dfs.append(split_df)
    
    df = pd.concat(split_dfs, ignore_index=True)

    return dict(data=df)

@conditional(lambda cfg: 'judge' in cfg.classification_config.metrics)
def prepare_judge_inputs(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    if cfg.llm_judge.judge_type not in ('classifier'):
       raise ValueError(f"Pipeline expected classifier as judge type, got: {cfg.llm_judge.judge_type}") 
    
    hydra_choices = HydraConfig.get().runtime.choices
    llm_judge_name = hydra_choices.get("llm_judge", None)

    if llm_judge_name!="best_setting_cls":
        warn_and_confirm(f"Classification is being called with best llm_judge settings. To use the best llm_judge settings, enter 'n' in the prompt that follows and re-run the script with override: +llm_judge=best_setting")

    icl_file = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.icl_file
    )

    with open(icl_file, 'r') as f:
        icl_examples = json.load(f)[cfg.llm_judge.icl_id]
    
    example_strings = {}

    for domain in icl_examples.keys():
        example_strings[domain] = ""
        for ex in icl_examples[domain]:
            if cfg.llm_judge.judge_type == 'rater':
                if cfg.llm_judge.label_space == [0,1] and ex['rating'] == 0.5:
                    continue
                else:
                    example_strings[domain]+= format_icl_examples(ex, cfg.llm_judge.judge_type, cfg.llm_judge.output_mode)
            else:
                example_strings[domain]+= format_icl_examples(ex, cfg.llm_judge.judge_type, cfg.llm_judge.output_mode)
    
    with open(os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.prompts_file
    ), "r") as f:
        judge_prompt = json.load(f)[cfg.llm_judge.prompt_id]['text']
    
    domain_wise_judge_prompts = {}

    for domain in example_strings.keys():
        domain_wise_judge_prompts[domain] = judge_prompt.format(
            domain = domain,
            icl_examples = example_strings[domain]
        )
    
    data['judge_prompt_template'] = data['domain'].map(domain_wise_judge_prompts)

    if data['judge_prompt_template'].isna().any():
        missing = data.loc[data['judge_prompt_template'].isna(), "domain"].unique()

        raise ValueError(f"Judge prompt template is missing for domains: {missing}")

    data['judge_prompt'] = (
        data['judge_prompt_template']
        + "Query: "
        + data['prompt']
    )

    data.drop(columns='judge_prompt_template', inplace=True)

    return dict(data=data)

@conditional(lambda cfg,data: ('judge_response' not in data.columns) and ('judge' in cfg.classification_config.metrics))
def get_responses_from_judge(cfg:config.main.ProjectConfig, data:pd.DataFrame):
    model = cfg.llm_judge.model.model_name
    backend = cfg.llm_judge.model.backend
    gen_kwargs = cfg.llm_judge.model.gen_kwargs

    inputs = data['judge_prompt'].tolist()

    responses = run_inference(backend, model, inputs, dict(cfg.llm_judge.model.engine_kwargs), **gen_kwargs)

    if responses is not None:
        data['judge_response'] = responses
        return dict(data=data)
    else:
        if backend in ('openai', 'claude'):
            return PIPELINE_ABORT
        else:
            raise ValueError("run_inference returned None")

@conditional(lambda cfg,data: (False if 'judge_label' in data else True) and ('judge' in cfg.classification_config.metrics))
def parse_judge_outputs(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    parse_func = partial(parse_output, parse_type=cfg.llm_judge.judge_type, output_mode=cfg.llm_judge.output_mode)

    parsed = data['judge_response'].apply(parse_func)

    if cfg.llm_judge.output_mode == 'label_only':
        data['judge_label'] = parsed
    elif cfg.llm_judge.output_mode == 'label_and_reasoning':
        data['judge_label'] = parsed.apply(lambda x: x.get('label') if isinstance(x,dict) else None)
        data['judge_reason'] = parsed.apply(lambda x:x.get('reason') if isinstance(x, dict) else None)
    else:
        raise ValueError(f"Unknown output_mode: {cfg.llm_judge.output_mode}")
    
    data["judge_label"] = data["judge_label"].str.lower().str.strip()

    return dict(data=data)

def compute_readability_metrics(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    hydra_choices = HydraConfig.get().runtime.choices
    cls_config_name = hydra_choices.get("classification_config")

    if cls_config_name!='best_setting':
         warn_and_confirm(f"Classification is being called with best llm_judge settings. To use the best classification settings, enter 'n' in the prompt that follows and re-run the script with override: +classification_config=best_setting")
    
    metrics = cfg.classification_config.metrics
    thresholds = cfg.classification_config.readability_thresholds
    agg_sch = cfg.classification_config.aggregation_scheme

    method_map = {
        'ari':textstat.automated_readability_index,
        'fkgl':textstat.flesch_kincaid_grade,
        'dale_chall':textstat.dale_chall_readability_score
    }
    
    for metric in metrics:
        if metric != "judge":
            data[f"{metric}_score"] = data["prompt"].apply(method_map[metric])
    
    for metric in metrics:
        if metric!= "judge":
            data[f"{metric}_label"] = data[f"{metric}_score"].apply(lambda s: get_label_from_readability_score(score=s, threshold=thresholds[metric]))
    
    pred_series = [data[f"{m}_label"] for m in metrics]
    
    data["prompt_type"] = pd.concat(pred_series, axis=1).apply(aggregate(agg_sch), axis=1)

    return dict(data=data)

def save_data(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    for split, group_df in data.groupby("split"):
        save_path = os.path.join(
            cfg.paths.data_dir,
            cfg.paths.benchmark_dir,
            f"{split}.csv"
        )

        group_df[["prompt_hash", "prompt", "domain", "prompt_type"]].to_csv(save_path)

def get_tasks(cfg:config.main.ProjectConfig):
    
    return [
        load_data,
        prepare_judge_inputs,
        get_responses_from_judge,
        parse_judge_outputs,
        compute_readability_metrics,
        save_data
    ]
