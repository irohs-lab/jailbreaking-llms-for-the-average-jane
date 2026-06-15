import src.jailbreaks as jailbreaks
from src.utils.inference import get_bert_embeddings
from src.utils.dr import reduce_dimensionality
from src.utils.tasks import conditional
import config.main
import os
import pandas as pd
import torch
import numpy as np
from rich import print as rprint
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


def setup_learn_bert_embeddings(cfg:config.main.ProjectConfig):
    rprint("[bold turquoise4] Learning BERT Embeddings [/bold turquoise4]")
    rprint(f"[bold cyan] Model: [/bold cyan][bold magenta]{cfg.cv_embedding.model_name}[/bold magenta]")
    
    if cfg.eval.jailbreak_set is not None:
        cfg.eval.jailbreaks = cfg.eval.jailbreak_set.jailbreaks
    hydra_choices = HydraConfig.get().runtime.choices
    jb_set_name = hydra_choices.get("eval/jailbreak_set", None)
    num_jb = len(cfg.eval.jailbreaks)

    if jb_set_name is not None:

        if jailbreaks == [None]:
            jailbreak_str = "baseline (1 jailbreak)"
        else:
            if num_jb == 1:
                jailbreak_str = f"{jb_set_name} ({num_jb} jailbreak)"
            else:
                jailbreak_str = f"{jb_set_name} ({num_jb} jailbreaks)"
        rprint(f"[bold cyan] Jailbreak Set: [/bold cyan][bold magenta]{jailbreak_str}[/bold magenta]")
    else:

        named = ["baseline" if jb is None else jb for jb in jailbreaks]

        if num_jb <= 3:
            jailbreak_str = f"{', '.join(named)} ({num_jb})"
        else:
            preview = ", ".join(named[:3])
            jailbreak_str = f"{preview}, … ({num_jb} jailbreaks)"
        
        rprint(f"[bold cyan] Jailbreaks: [/bold cyan][bold magenta]{jailbreak_str}[/bold magenta]")


def get_data(cfg:config.main.ProjectConfig):
    data_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.benchmark_dir,
    )
    data = {}
    for split in ('train', 'test', 'val'):
        data[split] = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
        data[split]['split'] = split
    
    _jailbreaks = ["baseline" if jb is None else jb for jb in cfg.eval.jailbreaks]

    jb_data = {}

    for split, split_df in data.items():
        jb_data[split] = {}

        for jb in _jailbreaks:
            if jb == 'baseline': jb_func = lambda x:x
            else: jb_func = getattr(jailbreaks, jb)

            jb_df = split_df.copy(deep=True)
            jb_df['jb_prompt'] = jb_df['prompt'].apply(jb_func)

            jb_data[split][jb] = jb_df

    
    return dict(data=data, jb_dfs=jb_data)

def obtain_bert_embeddings(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], jb_dfs:dict[str,dict[str, pd.DataFrame]]):

    save_parent_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.sent_emb_data_dir,
        cfg.cv_embedding.model_name.split('/')[-1].replace('-', '_')
    )

    jb_embeddings = {}
    for split, split_jb_dfs in jb_dfs.items():
        jb_embeddings[split] = {}
        cache_root = os.path.join(
            save_parent_dir,
            split
        )
        os.makedirs(cache_root, exist_ok=True)
        for jb, jb_df in tqdm(split_jb_dfs.items()):
            cache_path = os.path.join(cache_root, f"{jb}.pt")
            if not os.path.exists(cache_path):

                inputs = jb_df['jb_prompt'].tolist()
                phashes = jb_df['prompt_hash'].tolist()
                
                jb_bert_embedding = get_bert_embeddings(inputs, cfg.cv_embedding.model_name, batch_size=cfg.cv_embedding.batch_size)
                jb_embeddings[split][jb] = {
                    "embeddings": jb_bert_embedding,
                    "prompt_hashes": phashes
                }

                torch.save(jb_embeddings[split][jb], cache_path)
            else:
                jb_embeddings[split][jb] = torch.load(cache_path)
        
    return dict(data=data, jb_dfs=jb_dfs, jb_embeddings=jb_embeddings)

def get_tasks(cfg:config.main.ProjectConfig):
    
    return [
        setup_learn_bert_embeddings,
        get_data,
        obtain_bert_embeddings
    ]