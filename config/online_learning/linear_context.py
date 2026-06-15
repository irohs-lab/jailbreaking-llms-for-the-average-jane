import dataclasses
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class LinearContextualBandits(OnlineLearningConfig):
    
    name: str = "linear_cb"
    display_name:str = "Linear Contextual Bandits (Abe & Long, '99)"
    context_vector_model: str = "google/embeddinggemma-300m"
    dr_solver: str|None = None
    context_dim: int|None = None
    use_matryoshka:bool = False
    truncate_dim:int|None = None

store = hydra.core.config_store.ConfigStore.instance()
store.store("linear_cb", group="ol_scheme", node=LinearContextualBandits)

store.store(
    name="best_setting_linear_cb",
    group="ol_scheme",
    node=LinearContextualBandits(
        dr_solver="pca",
        context_dim=20,
        context_vector_model="answerdotai/ModernBERT-base"
    )
)