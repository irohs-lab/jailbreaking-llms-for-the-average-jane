import dataclasses
from config.model import ModelConfig
from enum import Enum

@dataclasses.dataclass
class LLMJudgeConfig:
    """
    Configuration class for the LLM judge system.

    Attributes:
        judge_type (str): Specifies the type of judge to use. Default is "rater". Options: "rater", "classifier", "enhancer", "decomposer"
        judge (ModelConfig): Configuration for the judge model. Defaults to a ModelConfig instance with model set to 'meta-llama/Llama-3.3-70b-Instruct'.
    """

    judge_type: str = "rater" # required for determining the output parser
    model: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig(model_name='meta-llama/Llama-3.3-70B-Instruct'))
    use_icl:bool = True

    prompt_id: str|None = None
    output_mode:str|None = None
    collapse_val_gt: bool = False
    icl_id: str|None = None