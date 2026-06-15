import config.main
from src.utils.tasks import conditional
import src.jailbreaks as jailbreaks

import pandas as pd
import os



def load_data(cfg: config.main.ProjectConfig):

    data_dir = os.path.join(cfg.paths.data_dir, cfg.paths.benchmark_dir)

    splits = []

    eval_splits = cfg.eval.splits

    for split in eval_splits:
        split_file = os.path.join(data_dir, f"{split}.csv")
        split_df = pd.read_csv(split_file)
        split_df['split'] = split
        splits.append(split_df)

    df = pd.concat(splits, ignore_index=True)

    return dict(data=df)


def apply_jailbreak(cfg: config.main.ProjectConfig, data:pd.DataFrame):
    
    dfs = []
    
    for jailbreak_func_name in cfg.eval.jailbreaks:
        if jailbreak_func_name is not None:
            jb_name = jailbreak_func_name
            jailbreak_func = getattr(jailbreaks, jailbreak_func_name)
        else:
            jailbreak_func = lambda x:x
            jb_name = 'baseline'
        
        jb_df = data.copy(deep=True)

        jb_df['jb_prompt'] = jb_df['prompt'].apply(jailbreak_func)
        jb_df['jailbreak'] = jb_name

        dfs.append(jb_df)
    
    data = pd.concat(dfs, ignore_index=True)

    return dict(data=data)