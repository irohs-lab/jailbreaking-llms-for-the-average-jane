import dataclasses

@dataclasses.dataclass
class OnlineLearningConfig:

    train_rounds: int|None = None
    prompt_type: str|None = None # set this to 'complex' or 'simple' to only use those prompt types
    normalize_cv:bool = True