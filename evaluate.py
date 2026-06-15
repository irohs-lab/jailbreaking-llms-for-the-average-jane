import hydra
import dotenv

import config.main
from src.utils import tasks
from src.pipeline import evaluation, online_learning, validation

@hydra.main(config_name="main", version_base=None)
def main(config: config.main.ProjectConfig):
    if config.eval.use_ol:
        pipeline = tasks.Pipeline(
            dotenv.load_dotenv,
            *online_learning.get_tasks(config),
            cfg=config
        )
    
    elif config.val:
        pipeline = tasks.Pipeline(
            dotenv.load_dotenv,
            *validation.get_tasks(config),
            cfg=config
        )
    else:
        pipeline = tasks.Pipeline(
            dotenv.load_dotenv,
            *evaluation.get_tasks(config),
            cfg=config
        )
    pipeline.run()

if __name__ == "__main__":
    main()