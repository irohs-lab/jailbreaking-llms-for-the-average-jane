import config.main
from config.model import get_model_config_name, ModelConfig
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_data(cfg:config.main.ProjectConfig):

    data_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.benchmark_dir,
    )

    splits = ('train', 'test', 'val')
    data = {}
    for split in splits:
        data[split] = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
        data[split]['split'] = split

    train_domains = cfg.attack.train_domains
    test_domains = cfg.attack.test_domains

    data['train'] = data['train'].loc[
        (data['train']['domain'].isin(train_domains))
    ]
    data['val']  = data['val'].loc[
        (data['val']['domain'].isin(train_domains))
    ]

    data['test'] = data['test'].loc[
        (data['test']['domain'].isin(test_domains))
    ]
    if cfg.ol_scheme.train_rounds is not None:
        counts = data['train']["domain"].value_counts(normalize=True)
        samples_per_group = (counts * cfg.ol_scheme.train_rounds).round().astype(int)

        data['train'] = (
            data['train'].groupby('domain', group_keys=False)
            .apply(lambda x:x.sample(n=samples_per_group[x.name], random_state=cfg.seed))
        )

    return dict(data=data)

def get_expert_data(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame]):

    cache_root = os.path.join(
        cfg.paths.expert_data_dir,
        cfg.paths.cache_dir,
        "target_model_responses"
    )

    target_model = cfg.eval.target_model.model_name.split('/')[-1].replace('-','_')
    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"
    judge_key = (
        get_model_config_name(cfg.llm_judge.model, not_hash=True)
        + f"__{cfg.llm_judge.prompt_id}"
        + f"__{cfg.llm_judge.output_mode}"
        + f"__{icl_tag}"
    )
    target_key = get_model_config_name(cfg.eval.target_model, not_hash=True)
    train_domains = cfg.attack.train_domains
    test_domains = cfg.attack.test_domains
    
    expert_data = {}
    experts = [exp if exp is not None else 'baseline' for exp in cfg.eval.jailbreaks]

    n = len(experts)

    total_steps = len(data) * len(experts)

    with tqdm(total=total_steps, desc="Loading expert data") as pbar:

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
                    
                    expert_df.loc[
                        expert_df['model_response'].isna(), rating_col
                    ] = 1.0

                    hash_rating_map.update(**dict(zip(expert_df['prompt_hash'], expert_df[rating_col])))
            
                ratings = split_df['prompt_hash'].map(hash_rating_map)
                expert_data[split][:, jb_idx] = ratings.to_numpy()
                pbar.update(1)

    # for split in expert_data:
    #     mask = ~np.isnan(expert_data[split]).any(axis=1)
    #     expert_data[split] = expert_data[split][mask]
    #     data[split] = data[split].iloc[mask].reset_index(drop=True)

    for split, split_df in data.items():
        print(f"Split: {split}, Split Size: {len(split_df)}")

    return dict(data=data, expert_data=expert_data)