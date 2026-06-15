import pathlib
import dataclasses
from typing import Callable, Optional, Any
from .base import LLMJudgeConfig
from config.model import ModelConfig

import hydra.core.config_store

@dataclasses.dataclass
class ClassifierLLMConfig(LLMJudgeConfig):

    val_set: pathlib.Path = dataclasses.field(default="cls_llm_val.csv")

    prompt_id: str|None = "cls_prompt_v1"
    output_mode:str|None = "label_only"
    label_space: list[str] = dataclasses.field(default_factory=lambda: ['simple', 'complex'])
    collapse_val_gt: bool = False
    icl_id: str|None = 'cls_prompt'


store = hydra.core.config_store.ConfigStore.instance()
store.store("classify", group="llm_judge", node=ClassifierLLMConfig)

store.store(
    name="best_setting_cls",
    group="llm_judge",
    node=ClassifierLLMConfig(
        prompt_id="cls_prompt_w_reasoning_v5",
        judge_type="classifier",
        model=ModelConfig(
            model_name="gpt-4.1-2025-04-14",
            backend="openai",
            gen_kwargs=dict(max_output_tokens=512)
        ),
        output_mode="label_and_reasoning",
        label_space=['simple', 'complex'],
        collapse_val_gt=False,
        icl_id='cls_prompt'
    )
)