import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class RandomizedWeightedMajority(OnlineLearningConfig):

    name:str = "randomized_weighted_majority"
    display_name:str = "Randomized Weighted Majority"
    lr: float|None = None

store = hydra.core.config_store.ConfigStore.instance()
store.store("rwm", group="ol_scheme", node=RandomizedWeightedMajority)