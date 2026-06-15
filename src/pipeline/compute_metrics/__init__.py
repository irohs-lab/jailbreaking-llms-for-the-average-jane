import config.main
from src.pipeline.compute_metrics.metrics import load_and_measure_asr, load_and_measure_baseline
from src.pipeline.compute_metrics.save import save_avg_asr, save_baseline_res

def get_tasks(cfg:config.main.ProjectConfig):
    tasks = [
        load_and_measure_asr,
        save_avg_asr,
        load_and_measure_baseline,
        save_baseline_res

    ]

    return tasks

__all__ = [
    "get_tasks"
]