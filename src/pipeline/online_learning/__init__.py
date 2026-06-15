import config.main
from config.model import get_model_config_name, ModelConfig
from src.pipeline.online_learning.data_utils import get_data, get_expert_data
from src.pipeline.online_learning.logging import get_comparator_metrics,log_results
from src.pipeline.online_learning.algorithms import learn_weights
from src.pipeline.online_learning.hyperparameters import get_ol_kwargs
from colorama import Fore, Style, init
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import numpy as np

init(autoreset=True)

def setup_evaluation(cfg:config.main.ProjectConfig):
    ol_dict = OmegaConf.to_container(cfg.ol_scheme, resolve=True)
    print(f"{Fore.GREEN}--- Online Learning Algorithm: {Fore.RED}{cfg.ol_scheme.display_name}{Fore.GREEN} ---{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Attack Mode{Style.RESET_ALL}: {Fore.YELLOW}{"Continual" if cfg.attack.continual else "Transfer"}{Style.RESET_ALL}")
    if not cfg.attack.continual and cfg.attack.num_passes > 1:
        print(f"{Fore.CYAN}Num Passes{Style.RESET_ALL}: {Fore.YELLOW}{cfg.attack.num_passes}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Train Domains{Style.RESET_ALL}: {Fore.YELLOW}{cfg.attack.train_domains}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Test Domains{Style.RESET_ALL}: {Fore.YELLOW}{cfg.attack.test_domains}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Target Model{Style.RESET_ALL}: {Fore.YELLOW}{cfg.eval.target_model.model_name}{Style.RESET_ALL}")
    for k, v in ol_dict.items():
        if k in ('experts', 'name'): continue
        print(f"{Fore.CYAN}{k}{Style.RESET_ALL}: {Fore.YELLOW}{v}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}--------------------------------{Style.RESET_ALL}")

    if cfg.eval.jailbreak_set is not None:
        cfg.eval.jailbreaks = cfg.eval.jailbreak_set.jailbreaks
    
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
    
    print(f"Jailbreak Set: {Fore.MAGENTA}{jb_set_name}({num_jb}){Style.RESET_ALL}")


def get_tasks(cfg: config.main.ProjectConfig):
    
    return [
        setup_evaluation,
        get_data,
        get_expert_data,
        get_ol_kwargs,
        learn_weights(cfg.ol_scheme.name),
        get_comparator_metrics,
        log_results
    ]