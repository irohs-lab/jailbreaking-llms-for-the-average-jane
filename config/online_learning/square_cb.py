import dataclasses
from typing import Dict, Any
import hydra.core.config_store
from .base import OnlineLearningConfig

@dataclasses.dataclass
class SquareCB(OnlineLearningConfig):

    name:str = 'square_cb'
    display_name:str = 'SquareCB (Foster & Rakhlin, 2020)'
    context_vector_model: str = 'google/embeddinggemma-300m'
    dr_solver: str|None = None
    context_dim: int|None = None
    use_matryoshka:bool = False
    truncate_dim:int|None = None
    regression_oracle: str = 'VovkOnlineRegressionOracle'
    failure_prob: float = 0.01 # to allow bounding infinity norm of context vectors by 1 in Theorem 1 of Vovk.

    online_regressor_params: Dict[str, Any] = dataclasses.field(default_factory= lambda: dict()) # default params set for Vovk Regressor. Change accordingly
    
def get_sqcb_name(cfg:SquareCB):
    if cfg.regression_oracle == 'VovkOnlineRegressionOracle':
        params = "_".join([str(v) for k,v in cfg.online_regressor_params.items()])
        return '_'.join([
            cfg.name,
            f"vovk_{params}"
        ])
    else: 
        return cfg.name

store = hydra.core.config_store.ConfigStore.instance()
store.store("square_cb", group="ol_scheme", node=SquareCB)

store.store(
    name="best_setting_square_cb",
    group="ol_scheme",
    node=SquareCB(
        dr_solver="rmap",
        context_dim=32,
        context_vector_model="google/embeddinggemma-300m"
    )
)