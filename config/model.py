import dataclasses
from typing import Dict, Any
import hashlib

@dataclasses.dataclass
class ModelConfig:
    model_name: str
    backend: str = "vllm"
    gen_kwargs: Dict[str,Any] = dataclasses.field(default_factory= lambda: dict())
    engine_kwargs: Dict[str, Any] = dataclasses.field(default_factory=lambda: dict(
        trust_remote_code= True,
        dtype='bfloat16',
        gpu_memory_utilization=0.9
    ))

def get_model_config_name(model_config:ModelConfig, not_hash=False):
    name = model_config.model_name.split('/')[-1].replace('-','_') + "_" + model_config.backend

    if len(model_config.gen_kwargs) <= 2 or not_hash:
        for k,v in model_config.gen_kwargs.items():
            name+=f"_{k}_{v}"
    else:
        for k,v in model_config.gen_kwargs.items():
            name+=f"_{k}_{v}"
            name = hashlib.md5(name.encode()).hexdigest()

    return name

    