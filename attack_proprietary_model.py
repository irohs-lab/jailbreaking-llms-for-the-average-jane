import hydra
import dotenv

import config.main
from src.utils import tasks
from src.pipeline import cross_model_transfer

@hydra.main(config_name="main", version_base=None)
def main(config:config.main.ProjectConfig):
    pipeline = tasks.Pipeline(
        dotenv.load_dotenv,
        *cross_model_transfer.get_tasks(config),
        cfg=config
    )

    pipeline.run()

if __name__ == "__main__":
    main()