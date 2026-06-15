import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class UniformPriors(OnlineLearningConfig):
    
    name: str = "uniform_priors"
    display_name:str = "Uniform Priors (Baseline)"

store = hydra.core.config_store.ConfigStore.instance()
store.store("uniform_priors", group="ol_scheme", node=UniformPriors)