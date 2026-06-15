import src.jailbreaks as jailbreaks
from src.utils.inference import get_sentence_embeddings, get_sentence_embeddings_vllm
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


def setup_learn_sentence_embeddings(cfg:config.main.ProjectConfig):
    rprint(f"[bold turquoise4] Learning {cfg.cv_embedding.name} Embeddings [/bold turquoise4]")
    rprint(f"[bold cyan] Model: [/bold cyan][bold magenta]{cfg.cv_embedding.model}[/bold magenta]")
    
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

def obtain_sentence_embeddings(cfg:config.main.ProjectConfig, data, jb_dfs):

    model_name = cfg.cv_embedding.model.split('/')[-1].replace('-', '_')

    save_parent_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.sent_emb_data_dir,
    )

    if cfg.cv_embedding.use_matryoshka and cfg.cv_embedding.truncate_dim is not None:
        save_parent_dir = os.path.join(
            save_parent_dir,
            f"{model_name}_matryoshka_{cfg.cv_embedding.truncate_dim}"
        )
    else:
        save_parent_dir = os.path.join(save_parent_dir, model_name)

    gen_kwargs = {}

    if cfg.cv_embedding.use_matryoshka and cfg.cv_embedding.truncate_dim is not None:
        gen_kwargs["truncate_dim"] = cfg.cv_embedding.truncate_dim

    jb_embeddings = {}

    prompts_to_encode = []
    mapping = []

    for split, split_jb_dfs in jb_dfs.items():

        jb_embeddings[split] = {}

        cache_root = os.path.join(save_parent_dir, split)
        os.makedirs(cache_root, exist_ok=True)

        for jb, jb_df in split_jb_dfs.items():

            cache_path = os.path.join(cache_root, f"{jb}.pt")

            if os.path.exists(cache_path):
                jb_embeddings[split][jb] = torch.load(cache_path)
                continue

            inputs = jb_df["jb_prompt"].tolist()
            phashes = jb_df["prompt_hash"].tolist()

            start_idx = len(prompts_to_encode)

            prompts_to_encode.extend(inputs)

            mapping.append({
                "split": split,
                "jb": jb,
                "start": start_idx,
                "end": start_idx + len(inputs),
                "hashes": phashes,
                "cache_path": cache_path
            })

    if len(prompts_to_encode) > 0:

        rprint(f"[bold cyan]Embedding {len(prompts_to_encode)} prompts with vLLM[/bold cyan]")

        all_embeddings = get_sentence_embeddings_vllm(
            prompts_to_encode,
            cfg.cv_embedding.model,
            **gen_kwargs
        )

        for m in mapping:

            split = m["split"]
            jb = m["jb"]

            emb = all_embeddings[m["start"]:m["end"]]

            obj = {
                "embeddings": emb,
                "prompt_hashes": m["hashes"]
            }

            jb_embeddings[split][jb] = obj

            torch.save(obj, m["cache_path"])

    return dict(data=data, jb_dfs=jb_dfs, jb_embeddings=jb_embeddings)



def get_tasks(cfg:config.main.ProjectConfig):
    
    return [
        setup_learn_sentence_embeddings,
        get_data,
        obtain_sentence_embeddings
    ]