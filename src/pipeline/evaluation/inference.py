import config.main
from config.model import get_model_config_name
from src.utils.tasks import conditional, PIPELINE_ABORT
from src.utils.inference import run_inference

import pandas as pd
import os


def gather_uncached_prompts(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    model = cfg.eval.target_model.model_name
    model_id = model.split('/')[-1].replace('-', '_')

    cache_dir = os.path.join(cfg.paths.data_dir, cfg.paths.cache_dir, "target_model_responses")

    cached_dfs, to_infer_dfs = [], []

    for (domain, jb, split), df_group in data.groupby(['domain', 'jailbreak', 'split']):

        cache_file = os.path.join(
            cache_dir,
            domain,
            jb,
            model_id,
            split,
            f"{get_model_config_name(cfg.eval.target_model, not_hash=True)}.csv"
        )

        if cfg.use_cache and os.path.exists(cache_file):
            cached_dfs.append(pd.read_csv(cache_file))
        else:
            to_infer_dfs.append(df_group)
    

    cached_df = (pd.concat(cached_dfs, ignore_index=True) if cached_dfs else None)
    to_infer_df = (pd.concat(to_infer_dfs, ignore_index=True) if to_infer_dfs else None)

    
    return dict(to_infer_df=to_infer_df)
            
@conditional(lambda to_infer_df: to_infer_df is not None and len(to_infer_df) > 0)
def run_target_inference(cfg:config.main.ProjectConfig, to_infer_df:pd.DataFrame):

    model = cfg.eval.target_model.model_name
    backend = cfg.eval.target_model.backend
    gen_kwargs = cfg.eval.target_model.gen_kwargs
    
    responses = run_inference(backend, model, to_infer_df['jb_prompt'].tolist(), dict(cfg.eval.target_model.engine_kwargs), **gen_kwargs)

    if responses is not None:
        to_infer_df['model_response'] = responses

        return dict(to_infer_df=to_infer_df)

    else: 
        if backend in ('openai', 'claude', 'grok'):
            return PIPELINE_ABORT
        else:
            raise ValueError(f"run_inference() returned None value")

@conditional(lambda to_infer_df: to_infer_df is not None)
def split_target_model_outputs(cfg:config.main.ProjectConfig, to_infer_df:pd.DataFrame):

    model = cfg.eval.target_model.model_name
    model_id = model.split('/')[-1].replace('-', '_')

    cache_dir = os.path.join(cfg.paths.data_dir, cfg.paths.cache_dir, "target_model_responses")

    for (domain, jb, split), df_group in to_infer_df.groupby(['domain', 'jailbreak', 'split']):

        cache_file = os.path.join(
            cache_dir,
            domain,
            jb,
            model_id,
            split,
            f"{get_model_config_name(cfg.eval.target_model, not_hash=True)}.csv"
        )

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        df_group.reset_index(drop=True, inplace=True)
        df_group.to_csv(cache_file, index=False)