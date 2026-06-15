import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class LinUCB(OnlineLearningConfig):

    name: str = 'linucb'
    display_name:str = 'LinUCB (Li et al., 2010)'
    context_vector_model: str = 'google/embeddinggemma-300m'
    dr_solver: str|None = None
    context_dim: int|None = None
    use_matryoshka:bool = False
    truncate_dim:int|None = None
    failure_prob:float = 0.01

store = hydra.core.config_store.ConfigStore.instance()
store.store("linucb", group="ol_scheme", node=LinUCB)

store.store(
    name="best_setting_linucb",
    group="ol_scheme",
    node=LinUCB(
        dr_solver="rmap",
        context_dim=10,
        context_vector_model="answerdotai/ModernBERT-base"
    )
)