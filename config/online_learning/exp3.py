import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class Exp3(OnlineLearningConfig):
    
    name:str = "exp3"
    display_name:str = "EXP3 (Auer et al., 2003)"
    lr: float|None = None

store = hydra.core.config_store.ConfigStore.instance()
store.store("exp3", group="ol_scheme", node=Exp3)