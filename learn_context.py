import hydra
import dotenv

import config.main
from src.utils import tasks
from src.pipeline import learn_bert_embeddings,learn_sentence_embeddings

@hydra.main(config_name="main", version_base=None)
def main(config: config.main.ProjectConfig):
    if config.cv_embedding.name == 'bert_embedding':
        pipeline = tasks.Pipeline(
            dotenv.load_dotenv,
            *learn_bert_embeddings.get_tasks(config),
            cfg=config
        )
    elif config.cv_embedding.name.endswith('_sentence_embedding'):
        pipeline = tasks.Pipeline(
            dotenv.load_dotenv,
            *learn_sentence_embeddings.get_tasks(config),
            cfg=config
        )
    else:
        raise NotImplementedError(f"Embedding scheme {config.cv_embedding.name} is not implemented yet for Context Vectors")
    
    pipeline.run()

if __name__ == "__main__":
    main()