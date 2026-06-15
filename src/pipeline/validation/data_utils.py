import config.main
from config.model import get_model_config_name
from src.utils.tasks import conditional

import pandas as pd
import os
import json
from collections import Counter

@conditional(lambda cfg: cfg.llm_judge.judge_type == 'classifier' and not os.path.exists(os.path.join(cfg.paths.data_dir, cfg.paths.val_data_dir, cfg.llm_judge.val_set)))
def build_val_data(cfg:config.main.ProjectConfig):

    cs_responses_file = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.crowdsource_data,
        cfg.paths.crowdsource_file
    )

    with open(cs_responses_file, "r") as f:
        cs_data = json.load(f)
    
    def map_label(label:str)->str:
        if label.lower() == 'yes':
            return 'simple'
        elif label.lower() == 'no':
            return 'complex'
        else:
            raise ValueError(f"Unknown label: {label}")
    
    def map_label_to_num(label:str)->str:
        if label == 'simple':
            return 0
        elif label == 'complex':
            return 1
        else:
            raise ValueError(f"Unknown label: {label}")
    
    
    rows = []
    
    for prompt_hash in cs_data:

        _responses = list(map(map_label, [x['Q1']['response'] for x in cs_data[prompt_hash]['annotations']]))

        # take majority vote
        aggregate_label = Counter(_responses).most_common(1)[0][0]

        rows.append({
            'prompt': cs_data[prompt_hash]['prompt_text'],
            'domain': cs_data[prompt_hash]['domain'],
            'label': aggregate_label
        })
    
    save_file = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.val_data_dir,
        cfg.llm_judge.val_set
    )

    pd.DataFrame(rows).to_csv(save_file, index=False)


def load_val_data(cfg:config.main.ProjectConfig):

    llm_judge = cfg.llm_judge
    judge_type = llm_judge.judge_type
    val_file = getattr(llm_judge, 'val_set', None)

    if val_file is None: raise ValueError(f"Validation File name not specified in judge config.")

    path = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.val_data_dir,
        val_file
    )

    data = pd.read_csv(path)

    if cfg.llm_judge.collapse_val_gt and cfg.llm_judge.judge_type == 'rater':
        data["rating"] = data["rating"].replace({0.5: 0.0})

    return dict(data=data)

def save_val_responses(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    data["judge_model"] = cfg.llm_judge.model.model_name.split('/')[-1]
    data["judge_backend"] = cfg.llm_judge.model.backend
    data["prompt_id"] = cfg.llm_judge.prompt_id
    data["output_mode"] = cfg.llm_judge.output_mode
    data["label_space"] = str(cfg.llm_judge.label_space)
    data["use_icl"] = cfg.llm_judge.use_icl

    save_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        cfg.paths.val_responses_dir,
        cfg.llm_judge.judge_type,
    )

    os.makedirs(save_dir, exist_ok=True)

    judge_id = get_model_config_name(cfg.llm_judge.model, not_hash=True)
    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"

    prompt_id = (
        f"_prompt_{cfg.llm_judge.prompt_id}"
        f"_out_{cfg.llm_judge.output_mode}"
        f"_labels_{''.join(map(str,cfg.llm_judge.label_space))}"
        f"_{icl_tag}"
    )

    save_file = os.path.join(save_dir, f'{judge_id}_{prompt_id}.csv')

    data.to_csv(save_file, index=False)

