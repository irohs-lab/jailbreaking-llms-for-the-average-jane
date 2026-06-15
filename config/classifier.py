from typing import Tuple, Dict, Optional
import dataclasses
import hydra.core.config_store

@dataclasses.dataclass
class ClassificationConfig:

    metrics:Tuple[str] = ('judge', 'dale_chall')
    readability_thresholds: Dict[str,float] = dataclasses.field(default_factory=lambda:dict(
        dale_chall=9.9
        )
    )
    aggregation_scheme:str = 'any'

store = hydra.core.config_store.ConfigStore.instance()

store.store(
    name="best_setting",
    group="classification_config",
    node=ClassificationConfig
)