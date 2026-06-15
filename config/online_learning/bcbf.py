import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class BudgetConstrainedBruteForce(OnlineLearningConfig):

    name:str = "bcbf"
    display_name:str = "Budget Constrained Brute Force (Baseline)"

store = hydra.core.config_store.ConfigStore.instance()
store.store("bcbf", group="ol_scheme", node=BudgetConstrainedBruteForce)