import dataclasses
import hydra.core.config_store
from .base import ContextVectorConfig

@dataclasses.dataclass
class SentenceEmbeddingConfig(ContextVectorConfig):

    model: str = 'google/embeddinggemma-300m'
    name: str = '${.model}_sentence_embedding'
    batch_size:int = 64
    truncate_dim:int|None = None
    use_matryoshka:bool=False


store = hydra.core.config_store.ConfigStore.instance()
store.store("sent_embedding", group="cv_embedding", node=SentenceEmbeddingConfig)