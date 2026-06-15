import pathlib
import dataclasses
from typing import Callable, Optional, Any, List, Dict
from enum import Enum

import hydra.core.config_store

from config.model import ModelConfig
from config.jb_sets import JailbreakSetConfig
from config.model_sets import ModelSetConfig
from .judge_tasks import LLMJudgeConfig
from .online_learning import OnlineLearningConfig, get_ol_algo_name
from .context_vector import ContextVectorConfig
from .classifier_val_sweep_config import ClassificationValSweepConfig
from .classifier import ClassificationConfig
from .online_learning.val_sweep.base import OLValSweepConfig

@dataclasses.dataclass
class PathsConfig:
    data_dir: pathlib.Path = "data"
    expert_data_dir:pathlib.Path = "data"
    cmt_cache_dir:pathlib.Path = "cross_model_transfer_cache"
    val_data_dir: pathlib.Path = "validation_data"
    val_responses_dir:pathlib.Path = "validation_responses"
    bert_data_dir: pathlib.Path = "bert_embeddings"
    sent_emb_data_dir: pathlib.Path = "sentence_embeddings"
    cls_emb_data_dir: pathlib.Path = "cls_embeddings"
    prompt_dir: pathlib.Path = "prompts"
    prompts_file: pathlib.Path = "prompts.json"
    icl_file: pathlib.Path = "icl_examples.json"
    benchmark_dir: pathlib.Path = "FrankensteinBench"
    result_dir: pathlib.Path = "results"
    cache_dir: pathlib.Path = "cache"
    log_dir: pathlib.Path = "logs"
    attack_save_dir: pathlib.Path = "attack_results"
    attack_log_dir: pathlib.Path = "attack_logs"
    crowdsource_data:pathlib.Path = "crowdsource_data"
    crowdsource_file:pathlib.Path = "crowdsourcing_responses.json"


@dataclasses.dataclass
class EvaluationConfig:

    metrics: list[str] = dataclasses.field(default_factory=lambda: ["asr"])

    jailbreaks: list[str | None] = dataclasses.field(default_factory=lambda: [None])

    jailbreak_set: JailbreakSetConfig | None = None

    splits: list[str] = dataclasses.field(default_factory=lambda:["val"])

    models_to_judge: list[str] = dataclasses.field(default_factory=lambda:["meta-llama/Llama-3.1-8B-Instruct"])

    target_model: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct"
    ))

    use_ol: bool = False

    eval_sv: bool = False

    run_inference:bool = False
    run_judge:bool = False # whether to judge the responses or not

    skip_judge_pass_sv: bool = False # set this to skip obtaining judge ratings for steering vector experiments

@dataclasses.dataclass
class AttackConfig:

    continual:bool = False
    train_domains: list[str] = dataclasses.field(
    default_factory=lambda: [
            'Finance',
            'Healthcare',
            'Legal',
            'Cybersecurity',
            'Education',
            'Public-harm'
        ]
    )

    test_domains: list[str] = dataclasses.field(
        default_factory=lambda: [
            'Finance',
            'Healthcare',
            'Legal',
            'Cybersecurity',
            'Education',
            'Public-harm'
        ]
    )
    num_passes:int = 1  
    uniform: bool = False # use uniform priors for attack

@dataclasses.dataclass
class MetricConfig:

    domains: list[str] = dataclasses.field(default_factory=lambda:["Finance", "Healthcare", "Education", "Cybersecurity", "Legal", "Public-harm"])

    model_set: ModelSetConfig|None = None

    eval_split: str = "test"


@dataclasses.dataclass
class CrossModelTransferConfig:
    
    target_model: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig)

    prior_model: ModelConfig = dataclasses.field(default_factory=lambda:ModelConfig)

    reuse_local_cache:bool = False # To re-use local cache when studying transfer b/w local models. 



@dataclasses.dataclass
class ProjectConfig:

    run: str | None = None # Unique identifier of an experiment
    seed: int = 18
    use_cache: bool = True

    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)

    eval: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)

    classifier_val_sweep_config: ClassificationValSweepConfig = dataclasses.field(default_factory=ClassificationValSweepConfig)

    classification_config: ClassificationConfig = dataclasses.field(default_factory=ClassificationConfig)

    val: bool = False

    summarize:bool = False

    llm_judge: LLMJudgeConfig|None = None

    ol_scheme: OnlineLearningConfig|None = None

    cv_embedding: ContextVectorConfig|None = None

    attack: AttackConfig = dataclasses.field(default_factory=AttackConfig)

    metric_config: MetricConfig = dataclasses.field(default_factory=MetricConfig)
    ol_val_sweep: OLValSweepConfig | None = None

    cmt_config: CrossModelTransferConfig = dataclasses.field(default_factory=CrossModelTransferConfig)


store = hydra.core.config_store.ConfigStore.instance()
store.store("main", ProjectConfig)

def get_ol_cfg_name(cfg:ProjectConfig):

    ol_key = get_ol_algo_name(cfg.ol_scheme)

    if cfg.ol_scheme.name == "randomized_weighted_majority":
        if cfg.ol_scheme.lr is not None:
            ol_key+=f"lr_{cfg.ol_scheme.lr}"
    elif cfg.ol_scheme.name == "exp3":
        if cfg.ol_scheme.lr is not None:
            ol_key+=f"lr_{cfg.ol_scheme.lr}"
    elif cfg.ol_scheme.name == "linear_cb":
        ol_key+=f"_embedding_model_{cfg.ol_scheme.context_vector_model.split('/')[-1].replace('-','_')}"

        if cfg.ol_scheme.use_matryoshka is not None and cfg.ol_scheme.truncate_dim is not None:
            ol_key+=f"_matryoshka_dim_{cfg.ol_scheme.truncate_dim}"

        if cfg.ol_scheme.dr_solver is not None and cfg.ol_scheme.context_dim is not None:
            ol_key+=f"_dr_solver_{cfg.ol_scheme.dr_solver}_dim_{cfg.ol_scheme.context_dim}"
    
    elif cfg.ol_scheme.name == "linucb":
        ol_key+=f"_failure_prob_{cfg.ol_scheme.failure_prob}"
        ol_key+=f"_embedding_model_{cfg.ol_scheme.context_vector_model.split('/')[-1].replace('-','_')}"

        if cfg.ol_scheme.use_matryoshka is not None and cfg.ol_scheme.truncate_dim is not None:
            ol_key+=f"_matryoshka_dim_{cfg.ol_scheme.truncate_dim}"

        if cfg.ol_scheme.dr_solver is not None and cfg.ol_scheme.context_dim is not None:
            ol_key+=f"_dr_solver_{cfg.ol_scheme.dr_solver}_dim_{cfg.ol_scheme.context_dim}"
    
    elif cfg.ol_scheme.name == "square_cb":
        
        ol_key+=f"_failure_prob_{cfg.ol_scheme.failure_prob}"
        ol_key+=f"_embedding_model_{cfg.ol_scheme.context_vector_model.split('/')[-1].replace('-','_')}"

        if cfg.ol_scheme.use_matryoshka is not None and cfg.ol_scheme.truncate_dim is not None:
            ol_key+=f"_matryoshka_dim_{cfg.ol_scheme.truncate_dim}"

        if cfg.ol_scheme.dr_solver is not None and cfg.ol_scheme.context_dim is not None:
            ol_key+=f"_dr_solver_{cfg.ol_scheme.dr_solver}_dim_{cfg.ol_scheme.context_dim}"
    
    elif cfg.ol_scheme.name == "uniform_priors":
        pass
    elif cfg.ol_scheme.name == "bcbf":
        pass
    elif cfg.ol_scheme.name == "thompson_sampling":
        pass
    else:
        raise ValueError(f"OL Algorithm Unknown: {cfg.ol_scheme.name}")
    
    if cfg.attack.num_passes > 1:
        ol_key += f"_passes_{cfg.attack.num_passes}"

    return ol_key
