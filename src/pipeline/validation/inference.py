import config.main
from config.model import get_model_config_name
from src.utils.tasks import conditional, PIPELINE_ABORT
from src.utils.inference import run_inference
from src.utils.parser import parse_output

import pandas as pd
import os
import json
from functools import partial

def output_parse_condition(cfg:config.main.ProjectConfig, data:pd.DataFrame):
    if cfg.llm_judge.judge_type == 'rater':
        if 'judge_rating' in data: return False
        else: return True
    elif cfg.llm_judge.judge_type == 'classifier':
        if 'judge_label' in data: return False
        else: return True
    else:
        raise NotImplementedError(f"output_parse_condition not implemented for judge type: {cfg.llm_judge.judge_type}")

def format_icl_examples(example:dict, judge_type:str, output_mode:str):
    if judge_type == 'rater':
        if output_mode == "rating_only":
            return f"Query: {example['query']}\nResponse: {example['response']}\nRating: {example['rating']}\n\n"
        else:
            return f"Query: {example['query']}\nResponse: {example['response']}\nRating: {example['rating']}\nReason: {example['reason']}\n\n"
    
    elif judge_type == 'classifier':

        if output_mode == "label_only":
            return f"Query: {example['query']}\n Label: {example['label']}\n\n"
        else:
            return f"Query: {example['query']}\n Label: {example['label']}\nReason: {example['reason']}\n\n"

def prepare_judge_inputs(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    if cfg.llm_judge.judge_type not in ('rater', 'classifier'):
        raise NotImplementedError(f"Validation pipeline is not setup for judge type: {cfg.llm_judge.judge_type}")
    
    judge_id = get_model_config_name(cfg.llm_judge.model, not_hash=True)
    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"

    prompt_id = (
        f"_prompt_{cfg.llm_judge.prompt_id}"
        f"_out_{cfg.llm_judge.output_mode}"
        f"_labels_{''.join(map(str,cfg.llm_judge.label_space))}"
        f"_{icl_tag}"
    )

    cache_path = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        cfg.paths.val_responses_dir,
        cfg.llm_judge.judge_type,
        f'{judge_id}_{prompt_id}.csv'
    )

    if os.path.exists(cache_path):
        data = pd.read_csv(cache_path)
        return dict(data=data)
    

    icl_file = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.icl_file 
    )

    example_strings = {}

    if cfg.llm_judge.use_icl:
        with open(icl_file, 'r') as f:
            icl_examples = json.load(f)[cfg.llm_judge.icl_id]

        for domain in icl_examples.keys():
            example_strings[domain] = ""
            for ex in icl_examples[domain]:
                if cfg.llm_judge.judge_type == 'rater':
                    if cfg.llm_judge.label_space == [0,1] and ex['rating'] == 0.5:
                        continue
                example_strings[domain] += format_icl_examples(
                    ex,
                    cfg.llm_judge.judge_type,
                    cfg.llm_judge.output_mode
                )
    else:
        for domain in data["domain"].unique():
            example_strings[domain] = ""
    
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
    
    if cfg.llm_judge.judge_type == 'rater':
        data['judge_prompt'] = (
            data['judge_prompt_template']
            + 'Query: '
            + data['prompt']
            + '\nResponse:'
            + data['response']
        )
    elif cfg.llm_judge.judge_type == 'classifier':
        data['judge_prompt'] = (
            data['judge_prompt_template']
            + "Query: "
            + data['prompt']
        )
    else:
        pass

    data.drop(columns='judge_prompt_template', inplace=True)

    return dict(data=data)

@conditional(lambda data: 'judge_response' not in data.columns)
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

@conditional(output_parse_condition)    
def parse_judge_outputs(cfg:config.main.ProjectConfig, data:pd.DataFrame):

    parse_func = partial(parse_output, parse_type=cfg.llm_judge.judge_type, output_mode=cfg.llm_judge.output_mode)

    parsed = data['judge_response'].apply(parse_func)

    if cfg.llm_judge.judge_type == 'rater':
        if cfg.llm_judge.output_mode == "rating_only":
            data['judge_rating'] = pd.to_numeric(parsed, errors='coerce')
        
        elif cfg.llm_judge.output_mode == "rating_and_reasoning":
            data['judge_rating'] = pd.to_numeric(
                parsed.apply(lambda x: x.get("rating") if isinstance(x, dict) else float("nan")),
                errors='coerce'
            )

            data['judge_reason'] = parsed.apply(
                lambda x: x.get("reason") if isinstance(x, dict) else None
            )
        else:
            raise ValueError(f"Unknown output_mode: {cfg.llm_judge.output_mode}")
    elif cfg.llm_judge.judge_type == 'classifier':
        if cfg.llm_judge.output_mode == 'label_only':
            data['judge_label'] = parsed
        elif cfg.llm_judge.output_mode == 'label_and_reasoning':
            data['judge_label'] = parsed.apply(lambda x: x.get('label') if isinstance(x,dict) else None)
            data['judge_reason'] = parsed.apply(lambda x:x.get('reason') if isinstance(x, dict) else None)
        else:
            raise ValueError(f"Unknown output_mode: {cfg.llm_judge.output_mode}")

    return dict(data=data)  