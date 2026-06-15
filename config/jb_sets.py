import dataclasses
from typing import List, Optional
import hydra.core.config_store
from src.jailbreaks import jailbreaks
import pandas as pd
import numpy as np

@dataclasses.dataclass
class JailbreakSetConfig:
    jailbreaks: List[Optional[str]]


store = hydra.core.config_store.ConfigStore.instance()

store.store(
    name="full",
    group="eval/jailbreak_set",
    node=JailbreakSetConfig(
        jailbreaks=[
            None,
            *jailbreaks
        ]
    ),
)

store.store(
    name="short_context",
    group="eval/jailbreak_set",
    node=JailbreakSetConfig(
        jailbreaks=[
            None,
            *[jb for jb in jailbreaks if jb not in ('many_shot', 'binary_encode')]
        ]
    )
)

store.store(
    name="debug",
    group="eval/jailbreak_set",
    node=JailbreakSetConfig(
        jailbreaks=["refusal_suppression"]
    )
)

store.store(
    name="baseline",
    group="eval/jailbreak_set",
    node=JailbreakSetConfig(
        jailbreaks=[None]
    )
)

df = pd.read_csv('average_asr_over_domains_results_full.csv', index_col=0) # Load the output of compute_metrics

df_sorted = df.assign(average=lambda row: row.mean(axis=1)).sort_values('average', ascending=False).drop('average', axis=1)

for k in (1, 2, 5, 10, 15, 20, 30, 40, 50, 60):
    pruned_df = df_sorted.iloc[k:]
    top_k_pruned_jbs = pruned_df.index.to_list()

    store.store(
        name=f"top_{k}_pruned",
        group="eval/jailbreak_set",
        node=JailbreakSetConfig(
            jailbreaks=top_k_pruned_jbs
        )
    )

means = df.mean(axis=1)
good_arms = means >= 0.60
good_df = df[good_arms]
bad_df = df[~good_arms]
bad_jbs = bad_df.index.to_list()

for num_good in (1,2,3,4,5,7,9,10,12,15):
    rng = np.random.default_rng(seed=18)

    assert len(good_df) >= num_good, "Not enough good jailbreaks for requested n_good"
    
    good_idx = rng.choice(good_df.index, size=num_good, replace=False)
    good_sample = good_df.loc[good_idx]

    good_jbs = good_sample.index.to_list()

    final_jb_list = good_jbs + bad_jbs

    store.store(
        name=f"only_{num_good}_good_jbs",
        group="eval/jailbreak_set",
        node=JailbreakSetConfig(
            jailbreaks=final_jb_list
        )
    )

store.store(
    name=f"only_0_good_jbs",
    group="eval/jailbreak_set",
    node=JailbreakSetConfig(
        jailbreaks=bad_jbs
    )
)
