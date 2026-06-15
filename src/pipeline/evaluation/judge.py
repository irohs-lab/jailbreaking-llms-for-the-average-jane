import config.main
from config.model import get_model_config_name
import pandas as pd
from functools import partial
from src.utils.inference import run_inference
from src.utils.tasks import conditional, PIPELINE_ABORT
from src.utils.parser import parse_output
import json
import os
from hydra.core.hydra_config import HydraConfig


def gather_unjudged_outputs(cfg:config.main.ProjectConfig):

    response_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        "target_model_responses"
    )

    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"

    judge_id = (
        get_model_config_name(cfg.llm_judge.model, not_hash=True)
        + f"__{cfg.llm_judge.prompt_id}"
        + f"__{cfg.llm_judge.output_mode}"
        + f"__{icl_tag}"
    )
    domains = os.listdir(response_dir)

    hydra_choices = HydraConfig.get().runtime.choices
    jb_set_name = hydra_choices.get("eval/jailbreak_set", None)

    if len(cfg.eval.jailbreaks) == 0: 
        filter_jbs = False 
        print(f"Judging on {len(cfg.eval.jailbreaks)} Jailbreaks")
        if jb_set_name:
            print(f"Jailbreak Set: {jb_set_name}")
    else: 
        filter_jbs = True

    jailbreaks_to_judge = [
        "baseline" if jb is None else jb
        for jb in cfg.eval.jailbreaks
    ]
    
    if len(cfg.eval.models_to_judge) > 0 :
        filter_models = True
    else:
        filter_models = False

    models_to_judge = [m.split('/')[-1].replace('-','_') for m in cfg.eval.models_to_judge]

    if filter_models:
        print("Judging Models:")
        for idx, _m in enumerate(models_to_judge):
            print(f"{idx}. {_m}")

    unjudged_response_dfs = []

    for domain in domains:
        jailbreaks = os.listdir(os.path.join(response_dir, domain))

        for jb in jailbreaks:
            if filter_jbs and jb not in jailbreaks_to_judge:
                continue
            target_models = os.listdir(os.path.join(response_dir, domain, jb))

            for target_model in target_models:
                if filter_models and target_model not in models_to_judge:
                    continue
                splits = os.listdir(os.path.join(response_dir, domain, jb, target_model))

                for split in splits:
                    response_path = os.path.join(
                        response_dir,
                        domain,
                        jb,
                        target_model,
                        split
                    )

                    response_files = [
                        f for f in os.listdir(response_path)
                        if f.endswith(".csv") and "__judged_" not in f
                    ]
                    for response_file in response_files:
                        judged_file = response_file.replace(
                            ".csv",
                            f"__judged_{judge_id}.csv"
                        )
                        judged_file_path = os.path.join(
                            response_path,
                            judged_file
                        )

                        response_file_path = os.path.join(
                            response_path,
                            response_file
                        )

                        if os.path.exists(judged_file_path):
                            continue
                        else:
                            df = pd.read_csv(response_file_path)
                            df["_target_model"] = target_model
                            df["_domain"] = domain
                            df["_jailbreak"] = jb
                            df["_response_file"] = response_file
                            df["_split"] = split
                            unjudged_response_dfs.append(df)
    
    if not unjudged_response_dfs:
        print("No unjudged responses found.")
        return dict(collated_df=None)

    collated_df = pd.concat(unjudged_response_dfs, ignore_index=True)
    return dict(collated_df=collated_df)

@conditional(lambda collated_df: collated_df is not None)
def prepare_judge_inputs(cfg:config.main.ProjectConfig, collated_df:pd.DataFrame):
  
    icl_file = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.icl_file
    )
    
    example_strings = {}

    if cfg.llm_judge.use_icl:
        with open(icl_file, "r") as f:
            icl_examples = json.load(f)[cfg.llm_judge.icl_id]

        for domain in icl_examples.keys():
            example_strings[domain] = ""
            for ex in icl_examples[domain]:
                if cfg.llm_judge.label_space == [0,1] and ex['rating'] == 0.5:
                    continue
                if cfg.llm_judge.output_mode == "rating_only":
                    example_strings[domain] += (
                        f"Query: {ex['query']}\n"
                        f"Response: {ex['response']}\n"
                        f"Rating: {ex['rating']}\n\n"
                    )
                else:
                    example_strings[domain] += (
                        f"Query: {ex['query']}\n"
                        f"Response: {ex['response']}\n"
                        f"Rating: {ex['rating']}\n"
                        f"Reason: {ex['reason']}\n\n"
                    )
    else:
        for domain in collated_df["_domain"].unique():
            example_strings[domain] = ""
    
    
    with open(os.path.join(
        cfg.paths.data_dir,
        cfg.paths.prompt_dir,
        cfg.paths.prompts_file
    ), "r") as f:
        judge_prompt = json.load(f)[cfg.llm_judge.prompt_id]['text']
    
    domain_wise_judge_prompts = {}

    for domain in example_strings.keys():
        if cfg.llm_judge.use_icl:
            domain_wise_judge_prompts[domain] = judge_prompt.format(domain=domain, icl_examples=example_strings[domain])
        else:
            domain_wise_judge_prompts[domain] = judge_prompt.format(domain=domain)
    
    collated_df["judge_prompt_template"] = collated_df["_domain"].map(
    domain_wise_judge_prompts
    )

    if collated_df["judge_prompt_template"].isna().any():
        missing = collated_df.loc[
            collated_df["judge_prompt_template"].isna(), "_domain"
        ].unique()
        raise ValueError(f"Missing judge prompt templates for domains: {missing}")
    
    collated_df["judge_prompt"] = (
        collated_df["judge_prompt_template"]
        + "Query: "
        + collated_df["prompt"]
        + "\nResponse:"
        + collated_df["model_response"].fillna("")
    )

    collated_df.drop(columns="judge_prompt_template", inplace=True)
    
    return dict(collated_df=collated_df)

@conditional(lambda collated_df: collated_df is not None)
def get_ratings_from_judge(cfg:config.main.ProjectConfig, collated_df:pd.DataFrame):

    model = cfg.llm_judge.model.model_name
    backend = cfg.llm_judge.model.backend
    gen_kwargs = cfg.llm_judge.model.gen_kwargs

    inputs = collated_df['judge_prompt'].tolist()

    responses = run_inference(backend, model, inputs, dict(cfg.llm_judge.model.engine_kwargs), **gen_kwargs)

    if responses is not None:
        collated_df['judge_response'] = responses

        return dict(collated_df=collated_df)
    else:
        if backend in ('openai', 'claude'):
            return PIPELINE_ABORT
        else:
            raise ValueError("run_inference returned None")

@conditional(lambda collated_df: collated_df is not None)
def parse_judge_outputs(cfg:config.main.ProjectConfig, collated_df:pd.DataFrame):

    parse_func = partial(parse_output, parse_type=cfg.llm_judge.judge_type, output_mode=cfg.llm_judge.output_mode)

    collated_df['judge_rating'] = collated_df['judge_response'].apply(parse_func)

    collated_df['judge_rating'] = pd.to_numeric(collated_df['judge_rating'], errors='coerce')

    return dict(collated_df=collated_df)

@conditional(lambda collated_df: collated_df is not None)
def split_judge_outputs(cfg:config.main.ProjectConfig, collated_df:pd.DataFrame):

    base_save_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        "target_model_responses"
    )

    icl_tag = "icl" if cfg.llm_judge.use_icl else "noicl"

    judge_id = (
        get_model_config_name(cfg.llm_judge.model, not_hash=True)
        + f"__{cfg.llm_judge.prompt_id}"
        + f"__{cfg.llm_judge.output_mode}"
        + f"__{icl_tag}"
    )

    identifying_cols = [
        "_domain",
        "_target_model",
        "_jailbreak",
        "_response_file",
        "_split"
    ]

    for (domain, target_model, jailbreak, response_file, split), df_group in (collated_df.groupby(identifying_cols)):

        output_dir = os.path.join(
            base_save_dir,
            domain,
            jailbreak,
            target_model,
            split
        )

        os.makedirs(output_dir, exist_ok=True)

        judged_filename = response_file.replace(
            ".csv",
            f"__judged_{judge_id}.csv"
        )

        output_path = os.path.join(output_dir, judged_filename)

        df_out = df_group.drop(columns=identifying_cols)

        df_out = df_out.reset_index(drop=True)

        df_out.to_csv(output_path, index=False)