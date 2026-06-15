import hydra
import dotenv

import config.main
from src.utils import tasks
from src.pipeline import compute_metrics

@hydra.main(config_name="main", version_base=None)
def main(config:config.main.ProjectConfig):

    pipeline = tasks.Pipeline(
        dotenv.load_dotenv,
        *compute_metrics.get_tasks(config),
        cfg=config
    )

    pipeline.run()

if __name__=="__main__":
    main()