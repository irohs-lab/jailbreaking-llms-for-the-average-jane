import config.main
from src.pipeline.evaluation.data_utils import load_data, apply_jailbreak
from src.pipeline.evaluation.inference import gather_uncached_prompts, run_target_inference, split_target_model_outputs
from src.pipeline.evaluation.judge import gather_unjudged_outputs, prepare_judge_inputs, get_ratings_from_judge, parse_judge_outputs,split_judge_outputs

from rich.console import Console
from rich.table import Table
from hydra.core.hydra_config import HydraConfig


def setup_evaluation(cfg: config.main.ProjectConfig):

    if cfg.eval.jailbreak_set is not None:
        cfg.eval.jailbreaks = cfg.eval.jailbreak_set.jailbreaks
    console = Console()

    table = Table(show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Target Model", cfg.eval.target_model.model_name)
    if cfg.eval.run_judge:
        table.add_row("Rater Judge", cfg.llm_judge.model.model_name)
    jailbreaks = cfg.eval.jailbreaks
    num_jb = len(jailbreaks)

    hydra_choices = HydraConfig.get().runtime.choices
    jb_set_name = hydra_choices.get("eval/jailbreak_set", None)

    if jb_set_name is not None:

        if jailbreaks == [None]:
            jailbreak_str = "baseline (1 jailbreak)"
        else:
            if num_jb == 1:
                jailbreak_str = f"{jb_set_name} ({num_jb} jailbreak)"
            else:
                jailbreak_str = f"{jb_set_name} ({num_jb} jailbreaks)"
    else:

        named = ["baseline" if jb is None else jb for jb in jailbreaks]

        if num_jb <= 3:
            jailbreak_str = f"{', '.join(named)} ({num_jb})"
        else:
            preview = ", ".join(named[:3])
            jailbreak_str = f"{preview}, … ({num_jb} jailbreaks)"

    table.add_row("Jailbreaks", jailbreak_str)

    console.print(table)


def _get_inference_tasks(cfg:config.main.ProjectConfig):
    return [
        load_data,
        apply_jailbreak,
        gather_uncached_prompts,
        run_target_inference,
        split_target_model_outputs
    ]

def _get_judge_tasks(cfg:config.main.ProjectConfig):
    return [
        gather_unjudged_outputs,
        prepare_judge_inputs,
        get_ratings_from_judge,
        parse_judge_outputs,
        split_judge_outputs
        
    ]

def get_tasks(cfg: config.main.ProjectConfig):
    tasks = [setup_evaluation]

    if cfg.eval.run_inference:
        tasks.extend(_get_inference_tasks(cfg))
    
    if cfg.eval.run_judge:
        tasks.extend(_get_judge_tasks(cfg))

    return tasks

__all__ = [
    "get_tasks"
]