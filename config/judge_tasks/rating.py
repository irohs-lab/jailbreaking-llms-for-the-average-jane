import pathlib
import dataclasses
from typing import Callable, Optional, Any
from .base import LLMJudgeConfig

import hydra.core.config_store

@dataclasses.dataclass
class RaterLLMConfig(LLMJudgeConfig):

    prompt_id: Optional[str] = dataclasses.field(default="rating_prompt_binary_w_reasoning_v1")
    val_set: pathlib.Path = dataclasses.field(default="rater_llm_val.csv")
    output_mode:Optional[str] = dataclasses.field(default="rating_and_reasoning")
    label_space: list[float] = dataclasses.field(default_factory=lambda: [0.0,1.0])
    collapse_val_gt: bool = False
    icl_id: Optional[str] = dataclasses.field(default="rating_prompt")

store = hydra.core.config_store.ConfigStore.instance()
store.store("rate", group="llm_judge", node=RaterLLMConfig)
