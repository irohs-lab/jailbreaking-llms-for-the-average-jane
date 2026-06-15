from typing import List, Dict, Optional
import dataclasses
import hydra.core.config_store

@dataclasses.dataclass
class ClassificationValSweepConfig:

    metrics:List[str] = dataclasses.field(default_factory=lambda: ['judge', 'ari', 'fkgl', 'dale_chall'])
    readability_thresholds:Dict[str,List[float]] = dataclasses.field(default_factory=lambda:dict(
        ari=[13.0],
        fkgl=[12.0],
        dale_chall=[9.0]
    ))
    aggregation_schemes:List[str] = dataclasses.field(default_factory=lambda:['any', 'majority_vote']) # this can be 'any' or 'majority'

store = hydra.core.config_store.ConfigStore.instance()

store.store(
    name="full_sweep",
    group="classifier_val_sweep_config",
    node=ClassificationValSweepConfig(
        metrics=['judge', 'ari', 'fkgl', 'dale_chall'],
        readability_thresholds={
            'ari': list(range(14,26)),
            'fkgl': list(range(12,24)),
            'dale_chall': [9.0, 9.2, 9.4, 9.6, 9.8, 9.9]
        },
        aggregation_schemes=["any", "majority_vote"]
    )

)