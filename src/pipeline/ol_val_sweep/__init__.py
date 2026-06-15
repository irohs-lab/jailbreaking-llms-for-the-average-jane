import config.main

from src.pipeline.online_learning.data_utils import get_data
from .grid import generate_param_grid
from .runner import run_sweep_parallel
from .logging import select_best_params, write_best_config

def setup_val_sweep(cfg:config.main.ProjectConfig):
    if cfg.eval.jailbreak_set is not None:
        cfg.eval.jailbreaks = cfg.eval.jailbreak_set.jailbreaks

def get_tasks(cfg: config.main.ProjectConfig):

    return [
        setup_val_sweep,
        get_data,
        generate_param_grid,
        run_sweep_parallel,
        select_best_params,
        write_best_config
    ]