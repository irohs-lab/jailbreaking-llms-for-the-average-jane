import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class ThompsonSampling(OnlineLearningConfig):
    
    name:str = "thompson_sampling"
    display_name:str = "Thompson Sampling (Thompson, 1933)"

store = hydra.core.config_store.ConfigStore.instance()
store.store("thompson_sampling", group="ol_scheme", node=ThompsonSampling)