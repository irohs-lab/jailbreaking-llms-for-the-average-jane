import config.main
from src.utils.tasks import conditional

import textstat
import pandas as pd
import os
from functools import partial
from itertools import product, combinations
from collections import Counter


def get_label_from_readability_score(score:float, threshold:float):
    if score < threshold: return 'simple'
    else: return 'complex'

def aggregate(method:str):
    def aggregate_any(labels:pd.Series):
        _labels = labels.tolist()
        if 'simple' in _labels: return 'simple'
        else: return 'complex'
    
    def aggregate_mv(labels:pd.Series):
        labels = labels.tolist()
        return "simple" if labels.count("simple") >= labels.count("complex") else "complex"

    if method == 'any':
        return aggregate_any
    elif method == 'majority_vote':
        return aggregate_mv
    else:
        raise ValueError(f"Unknown method: {method}")

@conditional(lambda cfg: cfg.llm_judge.judge_type == "classifier")
def get_validation_settings(cfg:config.main.ProjectConfig, data:pd.DataFrame):
    classifier_config = cfg.classifier_val_sweep_config
    readability_thresholds = classifier_config.readability_thresholds
    metrics = classifier_config.metrics
    aggregation_schemes = classifier_config.aggregation_schemes

    settings = []
    setting_lists = {}

    for metric in metrics:
        if metric in readability_thresholds:
            setting_lists[metric] = []
            for t in readability_thresholds[metric]:
                setting_lists[metric].append((metric, t))
        else:
            setting_lists[metric] = [(metric,)]

    metric_combos=[]
    for r in range(1, len(setting_lists)+1):
        metric_combos.extend(list(combinations(setting_lists.keys(),r)))
    
    for combo in metric_combos:
        settings.extend(list(product(*[setting_lists[m] for m in combo], aggregation_schemes)))
    
    return dict(data=data, settings=settings, setting_lists=setting_lists)

@conditional(lambda cfg: cfg.llm_judge.judge_type == "classifier")
def compute_metric_based_labels(cfg:config.main.ProjectConfig, data:pd.DataFrame, settings:list[tuple|str]):

    data["judge_label"] = data["judge_label"].str.lower().str.strip()
    
    judge_id = data["judge_model"].iloc[0]
    prompt_id = data["prompt_id"].iloc[0]
    output_mode = data["output_mode"].iloc[0]

    sweep_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        "validation_sweeps",
        "classifier",
        judge_id,
    )

    os.makedirs(sweep_dir, exist_ok=True)

    sweep_path = os.path.join(
        sweep_dir,
        f"{prompt_id}__{output_mode}.parquet"
    )

    if os.path.exists(sweep_path):
        return dict(data=data)

    method_map = {
        'ari':textstat.automated_readability_index,
        'fkgl':textstat.flesch_kincaid_grade,
        'dale_chall':textstat.dale_chall_readability_score
    }
    
    for metric, func in method_map.items():
        col = f"{metric}_score"
        if col not in data.columns:
            data[col] = data["prompt"].apply(func)
    
    accuracies = []
    y_true = data['label'].str.lower()
    for setting in settings:
        *metrics, agg_scheme = setting
        metric_names = [m[0] for m in metrics]
        metric_part = "+".join(metric_names)
        thresholds = {}
        for m in metrics:
            if len(m)>1:
                thresholds[m[0]] = m[1]
        thresh_part = ",".join(
            f"{k}={v}" for k, v in thresholds.items()
        ) if thresholds else "—"
        setting_key=f"metrics={metric_part}|agg={agg_scheme}|thr={thresh_part}"

        metric_predictions = []
        for m_tuple in metrics:
            m = m_tuple[0]
            if m == "judge":
                metric_predictions.append(data['judge_label'])
            else:
                metric_predictions.append(data[f'{m}_score'].apply(lambda s:get_label_from_readability_score(score=s,threshold=m_tuple[1])))
        
        y_pred = pd.concat(metric_predictions, axis=1).apply(aggregate(agg_scheme), axis=1)
        acc = (y_true == y_pred).mean()

        accuracies.append({
            'setting_key':setting_key,
            'acc':acc
        })
    
    sweep_df = pd.DataFrame(accuracies)

    judge_id = data["judge_model"].iloc[0]
    prompt_id = data["prompt_id"].iloc[0]
    output_mode = data["output_mode"].iloc[0]

    sweep_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        "validation_sweeps",
        "classifier",
        judge_id,
    )

    os.makedirs(sweep_dir, exist_ok=True)

    sweep_path = os.path.join(
        sweep_dir,
        f"{prompt_id}__{output_mode}.parquet"
    )

    sweep_df.to_parquet(sweep_path, index=False)

    return dict(data=data)
    