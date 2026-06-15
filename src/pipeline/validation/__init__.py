import config.main
from src.pipeline.validation.data_utils import load_val_data, save_val_responses, build_val_data
from src.pipeline.validation.inference import prepare_judge_inputs, get_responses_from_judge, parse_judge_outputs
from src.pipeline.validation.metrics import get_validation_settings,compute_metric_based_labels
from src.pipeline.validation.summarize import fetch_validation_results, compute_diagnostics, display_summary

from rich.console import Console
import sys



def setup_validation_rater(cfg:config.main.ProjectConfig):
    console = Console()
    console.print(
        "[bold cyan]Judge Model: [/bold cyan]",
        cfg.llm_judge.model.model_name
    )
    console.print("[bold cyan]Prompt ID: [/bold cyan]", cfg.llm_judge.prompt_id)
    console.print("[bold cyan]Output Mode: [/bold cyan]",
                  cfg.llm_judge.output_mode)
    
    console.print("[bold cyan]Label Space: [/bold cyan]", cfg.llm_judge.label_space)

def setup_validation_cls(cfg:config.main.ProjectConfig):

    console = Console()
    console.print("[bold cyan]Classifier Judge Model:[/bold cyan]", cfg.llm_judge.model.model_name)
    console.print("[bold cyan]Prompt ID:[/bold cyan]", cfg.llm_judge.prompt_id)
    console.print("[bold cyan]Output Mode:[/bold cyan]", cfg.llm_judge.output_mode)
    console.print("[bold cyan]Label Space:[/bold cyan]", cfg.llm_judge.label_space)


def get_tasks_rater(cfg: config.main.ProjectConfig):
    
    tasks = [setup_validation_rater]

    if cfg.val:
        tasks.extend([
            load_val_data,
            prepare_judge_inputs,
            get_responses_from_judge,
            parse_judge_outputs,
            save_val_responses
        ])

    if cfg.summarize:
        tasks.extend([
            fetch_validation_results,
            compute_diagnostics(cfg),
            display_summary(cfg)
        ])

    return tasks

def get_tasks_classifier(cfg:config.main.ProjectConfig):

    tasks = [setup_validation_cls]

    if cfg.val:
        tasks.extend([
            build_val_data,
            load_val_data,
            prepare_judge_inputs,
            get_responses_from_judge,
            parse_judge_outputs,
            get_validation_settings,
            compute_metric_based_labels,
            save_val_responses

        ])
    if cfg.summarize:
        tasks.extend([
            fetch_validation_results,
            compute_diagnostics(cfg),
            display_summary(cfg)

        ])
    
    return tasks


def get_tasks(cfg: config.main.ProjectConfig):
    
    modules = sys.modules[__name__]
    try:
        get_tasks_func =  getattr(modules, f"get_tasks_{cfg.llm_judge.judge_type}")
    except AttributeError:
        raise ValueError(f"Unknown judge type: {cfg.llm_judge.judge_type}")
    
    return get_tasks_func(cfg)


__all__ = [
    "get_tasks"
]