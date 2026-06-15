import dataclasses
from typing import Dict, List, Any
import hydra.core.config_store


@dataclasses.dataclass
class SweepGroup:
    params: Dict[str, List[Any]]


@dataclasses.dataclass
class OLValSweepConfig:

    groups: List[SweepGroup] = dataclasses.field(default_factory=list)
    target_models: List[str] = dataclasses.field(default_factory=list)
    target_model_params:Dict = dataclasses.field(default_factory=dict)

    num_workers: int = 8
    cache_dir: str = "ol_val_sweep_cache"


store = hydra.core.config_store.ConfigStore.instance()

store.store(
    name="base",
    group="ol_val_sweep",
    node=OLValSweepConfig
)

store.store(
    name="embedding_sweep",
    group="ol_val_sweep",
    node=OLValSweepConfig(
        target_models= [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        target_model_params=dict(
            max_new_tokens=2048,
            backend="vllm"
        ),
        groups = [

        SweepGroup(
            params=dict(
                context_vector_model=[
                    "google/embeddinggemma-300m",
                    "answerdotai/ModernBERT-base"
                ],
                dr_solver=["pca", "diffred", "rmap"],
                context_dim=[10,20,32],
            )
        ),

        SweepGroup(
            params=dict(
                context_vector_model=[
                    "Qwen/Qwen3-Embedding-0.6B",
                    "Qwen/Qwen3-Embedding-4B",
                    "Qwen/Qwen3-Embedding-8B"
                ],
                use_matryoshka=[True],
                truncate_dim=[32],
            )
        ),
        SweepGroup(
            params=dict(
                context_vector_model=[
                    "Qwen/Qwen3-Embedding-0.6B",
                    "Qwen/Qwen3-Embedding-4B",
                    "Qwen/Qwen3-Embedding-8B"
                ],
                use_matryoshka=[True],
                truncate_dim=[32],
                dr_solver=["pca", "diffred", "rmap"],
                context_dim=[10,20]
            )
        )
    ]
    )
)

store.store(
    name="square_cb_embedding_sweep",
    group="ol_val_sweep",
    node=OLValSweepConfig(
        target_models= [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "Qwen/Qwen2.5-72B-Instruct"
        ],
        target_model_params=dict(
            max_new_tokens=2048,
            backend="vllm"
        ),
        groups = [

        SweepGroup(
            params=dict(
                context_vector_model=[
                    "google/embeddinggemma-300m",
                    "answerdotai/ModernBERT-base"
                ],
                dr_solver=["pca", "diffred", "rmap"],
                context_dim=[10,20,32],
            )
        ),

        SweepGroup(
            params=dict(
                context_vector_model=[
                    "Qwen/Qwen3-Embedding-0.6B",
                    "Qwen/Qwen3-Embedding-4B",
                    "Qwen/Qwen3-Embedding-8B"
                ],
                use_matryoshka=[True],
                truncate_dim=[32],
            )
        ),
        SweepGroup(
            params=dict(
                context_vector_model=[
                    "Qwen/Qwen3-Embedding-0.6B",
                    "Qwen/Qwen3-Embedding-4B",
                    "Qwen/Qwen3-Embedding-8B"
                ],
                use_matryoshka=[True],
                truncate_dim=[32],
                dr_solver=["pca", "diffred", "rmap"],
                context_dim=[10,20]
            )
        )
    ]
    )
)