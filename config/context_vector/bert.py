import dataclasses
import hydra.core.config_store
from .base import ContextVectorConfig

@dataclasses.dataclass
class BertEmbeddingConfig(ContextVectorConfig):

    name: str = "bert_embedding"
    model_name: str = 'answerdotai/ModernBERT-base'
    batch_size:int = 64


store = hydra.core.config_store.ConfigStore.instance()
store.store("bert", group="cv_embedding", node=BertEmbeddingConfig)