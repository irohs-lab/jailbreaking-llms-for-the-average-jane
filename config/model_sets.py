import dataclasses
from typing import List, Optional
import hydra.core.config_store

from config.model import ModelConfig


@dataclasses.dataclass
class ModelSetConfig:
    models: List[Optional[ModelConfig]]


model_list = [
    ("google/gemma-3-270m-it", "vllm", "max_new_tokens"),
    ("google/gemma-3-1b-it", "vllm", "max_new_tokens"),
    ("google/gemma-3-4b-it", "vllm", "max_new_tokens"),
    ("google/gemma-3-12b-it", "vllm", "max_new_tokens"),
    ("google/gemma-3-27b-it", "vllm", "max_new_tokens"),
    ("meta-llama/Llama-3.1-8B-Instruct", "vllm", "max_new_tokens"),
    ("meta-llama/Llama-3.1-70B-Instruct", "vllm", "max_new_tokens"),
    ("meta-llama/Llama-3.3-70B-Instruct", "vllm", "max_new_tokens"),
    ("openai/gpt-oss-20b", "vllm", "max_new_tokens"),
    ("openai/gpt-oss-120b", "vllm", "max_new_tokens"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "vllm", "max_new_tokens"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "vllm", "max_new_tokens"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "vllm", "max_new_tokens"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "vllm", "max_new_tokens"),
    ("Qwen/Qwen2.5-72B-Instruct", "vllm", "max_new_tokens")
]

store = hydra.core.config_store.ConfigStore.instance()

store.store(
    name="full",
    group="metric_config/model_set",
    node=ModelSetConfig(
        models=[
            ModelConfig(
                model_name=mname,
                backend = backend,
                gen_kwargs={
                    f"{max_tok_attr}": 2048
                }
            )
            for mname, backend, max_tok_attr in model_list
        ]
    )
)

store.store(
    name="debug",
    group="metric_config/model_set",
    node=ModelSetConfig(
        models=[
            ModelConfig(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                backend="vllm",
                gen_kwargs={
                    "max_new_tokens":2048
                }
            )
        ]
    )
)