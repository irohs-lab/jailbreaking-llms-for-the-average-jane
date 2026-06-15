import config.main
from config.model import get_model_config_name, ModelConfig
from config.main import get_ol_cfg_name
from src.utils.tasks import conditional
import pandas as pd
import numpy as np
import os
from rich import print as rprint
from hydra.core.hydra_config import HydraConfig


def nice_print(metric_name:str, metric:str):
    if not isinstance(metric, str):
        rprint(f"[bold cyan]{metric_name}:[/bold cyan] [magenta]{metric:0.2f}[/magenta]")
    else:
        rprint(f"[bold cyan]{metric_name}:[/bold cyan] [magenta]{metric}[/magenta]")

def get_comparator_metrics(cfg: config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], observed_asr:dict[str,np.float64], weight_log:dict[str,np.ndarray]):

    '''
    Transfer Mode: In this attack mode, we're interested in logging the following information:
        1. Train ASR
        2. Test ASR
        3. Best Expert in Hindsight
        4. Best Expert ASR (Train)
        5. Best Expert ASR (Test)
        6. Satisfaction (Train) = Train ASR - Best Expert ASR (Train)
        5. Satisfaction (Test) = Test ASR - Best Expert ASR (Test)
        6. Best possible ASR (Train)
        7. Best possible ASR (Test)
        8. Satisfaction Max (Train) = Best Possible ASR (Train) - Best Expert ASR (Train)
        9. Satisfaction Max (Test) = Best Possible ASR (Test) - Best Expert ASR (Test)
    
    Continual Mode: In this attack mode, the quantities of interest are:
    1. Train ASR
    2. Test ASR
    3. Best Expert in Hindsight (Train + Test)
    4. Best Expert Rate = Test ASR of Best Expert in Hindsight (Train + Test)
    5. Satisfaction = Observed ASR - Best Expert Rate
    6. Best Possible ASR
    7. Satisfaction Max = Best Possible ASR - Best Expert Rate

    '''
    if not cfg.attack.continual:

        T = expert_data['train'].shape[0]
        T_prime = T+expert_data['test'].shape[0]

        train_success_rates = 1 - expert_data['train'].mean(axis=0)
        best_expert_hindsight_index = int(train_success_rates.argmax())
        best_expert_hindsight = cfg.eval.jailbreaks[best_expert_hindsight_index]

        expert_success_rates = {
            'train': 1-expert_data['train'].mean(axis=0),
            'test': 1-expert_data['test'].mean(axis=0)
        }

        best_expert_asr = {
            'train': expert_success_rates['train'][best_expert_hindsight_index],
            'test': expert_success_rates['test'][best_expert_hindsight_index]
        }

        satisfaction = {
            'train': observed_asr['train'] - best_expert_asr['train'],
            'test': observed_asr['test'] - best_expert_asr['test']
        }

        best_possible_asr = {
            'train': np.mean(1-np.min(expert_data['train'], axis=1)),
            'test' : np.mean(1-np.min(expert_data['test'], axis=1))
        }

        satisfaction_max = {
            'train': best_possible_asr['train'] - best_expert_asr['train'],
            'test' : best_possible_asr['test'] - best_expert_asr['test']
        }

        nice_print("Training Rounds", T)
        nice_print("Test Rounds", T_prime-T)
        nice_print("Observed ASR (Train)", observed_asr['train'])
        nice_print("Observed ASR (Test)", observed_asr['test'])
        nice_print("Best Jailbreak in Hindsight", best_expert_hindsight)
        nice_print("Best Jailbreak Hindsight ASR (Train)", best_expert_asr['train'])
        nice_print("Best Jailbreak Hindsight ASR (Test)", best_expert_asr['test'])
        nice_print("Satisfaction (Train)", satisfaction['train'])
        nice_print("Satisfaction (Test)", satisfaction['test'])
        nice_print("Best Possible ASR (Train)", best_possible_asr['train'])
        nice_print("Best Possible ASR (Test)", best_possible_asr['test'])
        nice_print("Best Possible Satisfaction(Train)", satisfaction_max['train'])
        nice_print("Best Possible Satisfaction", satisfaction_max['test'])

        return dict(
            data=data,
            weight_log=weight_log,
            expert_data=expert_data,
            metrics_to_log=dict(
                observed_asr_train=observed_asr['train'],
                observed_asr_test=observed_asr['test'],
                best_jb_hindsight=best_expert_hindsight,
                best_jb_hs_asr_train=best_expert_asr['train'],
                best_jb_hs_asr_test=best_expert_asr['test'],
                satisfaction_train=satisfaction['train'],
                satisfaction_test=satisfaction['test'],
                best_possible_asr_train=best_possible_asr['train'],
                best_possible_asr_test=best_possible_asr['test'],
                best_possible_sat_train=satisfaction_max['train'],
                best_possible_sat_test=satisfaction_max['test']
            )
        )
    else:

        T = expert_data['train'].shape[0]
        T_prime = T+expert_data['test'].shape[0]

        full_expert_data = np.concatenate([expert_data['train'], expert_data['test']], axis=0)

        full_success_rates = 1-full_expert_data.mean(axis=0)
        best_expert_hindsight_index=int(full_success_rates.argmax())
        best_expert_hindsight=cfg.eval.jailbreaks[best_expert_hindsight_index]

        test_success_rates = 1-expert_data['test'].mean(axis=0)

        best_expert_asr_test = test_success_rates[best_expert_hindsight_index]

        best_expert_asr_full = full_success_rates[best_expert_hindsight_index]

        test_satisfaction = observed_asr['test'] - best_expert_asr_test

        full_observed_asr = (T*observed_asr['train'] + (T_prime-T)*observed_asr['test'])/T_prime

        full_satisfaction = full_observed_asr - best_expert_asr_full

        best_possible_asr_test = np.mean(1-np.min(expert_data['test'], axis=1))
        
        best_possible_asr_full = np.mean(1-np.min(full_expert_data, axis=1))

        satisfaction_max = {
            'test': best_possible_asr_test - best_expert_asr_test,
            'full': best_possible_asr_full - best_expert_asr_full
        }

        nice_print("Training Rounds", T)
        nice_print("Test Rounds", T_prime-T)
        nice_print("Train ASR", observed_asr['train'])
        nice_print("Test ASR", observed_asr['test'])
        nice_print("Best Expert in Hindsight (Train + Test)", best_expert_hindsight)
        nice_print("Test ASR of Best Hindsight Expert", best_expert_asr_test)
        nice_print("Full ASR of Best Hindsight Expert", best_expert_asr_full)
        nice_print("Test Satisfaction", test_satisfaction)
        nice_print("Full Satisfaction", full_satisfaction)
        nice_print("Best Possible ASR on test set", best_possible_asr_test)
        nice_print("Best Possible ASR on full", best_possible_asr_full)
        nice_print("Max Achievable Satisfaction on Test set", satisfaction_max['test'])
        nice_print("Max Achievable Satisfaction on Full", satisfaction_max['full'])

        return dict(
            data=data,
            weight_log=weight_log,
            expert_data=expert_data,
            metrics_to_log=dict(
                observed_asr_train=observed_asr['train'],
                observed_asr_test=observed_asr['test'],
                best_expert_hindsight=best_expert_hindsight,
                best_expert_asr_test=best_expert_asr_test,
                best_expert_asr_full=best_expert_asr_full,
                test_satisfaction=test_satisfaction,
                full_satisfaction=full_satisfaction,
                best_possible_asr_test=best_possible_asr_test,
                best_possible_asr_full=best_possible_asr_full,
                max_satisfaction_test=satisfaction_max['test'],
                max_satisfaction_full=satisfaction_max['full']
            )
        )


def log_results(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], weight_log:dict[str,np.ndarray], metrics_to_log):

    train_rounds = expert_data['train'].shape[0]
    test_rounds = expert_data['test'].shape[0]
    total_rounds = train_rounds+test_rounds

    num_experts = len(cfg.eval.jailbreaks)
    hydra_choices = HydraConfig.get().runtime.choices
    expert_set = hydra_choices.get("eval/jailbreak_set", None)
    domain_tag = 'traindomains_'
    for domain in cfg.attack.train_domains:
        domain_tag+=domain[0]
    
    domain_tag+='testdomains_'
    for domain in cfg.attack.test_domains:
        domain_tag+=domain[0]
    
    target_model=cfg.eval.target_model.model_name.split('/')[-1].replace('-', '_')

    ol_key = get_ol_cfg_name(cfg)

    save_root = os.path.join(
        cfg.paths.result_dir,
        "transfer" if not cfg.attack.continual else "continual",
        ol_key,
        domain_tag,
        target_model
    )

    os.makedirs(save_root, exist_ok=True)

    save_file = os.path.join(
        save_root,
        "result_file.csv"
    )

    log_metrics = {
        'train_rounds': train_rounds,
        'test_rounds': test_rounds,
        'total_rounds': total_rounds,
        'num_jailbreaks':num_experts,
        **({'jailbreak_set':expert_set} if expert_set is not None else {}),
        **{k:v for k,v in metrics_to_log.items() if k!='weight_log'}
    }

    primary_key = ['train_rounds', 'test_rounds', 'total_rounds', 'num_jailbreaks','jailbreak_set']
    if os.path.exists(save_file):
        df = pd.read_csv(save_file)
        df.drop_duplicates(inplace=True, subset=primary_key)
        rows = df.to_dict(orient='records')
        rows.append(log_metrics)
        df = pd.DataFrame(rows)
        df.drop_duplicates(inplace=True, subset=primary_key, keep='last')
        df.to_csv(save_file, index=False)
    else:
        df = pd.DataFrame([log_metrics])
        df.drop_duplicates(inplace=True, subset=primary_key)
        df.to_csv(save_file, index=False)

    if weight_log is not None:

        weight_log_file_train = os.path.join(
            save_root,
            f"trainRounds_{train_rounds}_testRounds_{test_rounds}_num_jailbreaks_{num_experts}{f'_expert_set_{expert_set}' if expert_set is not None else ''}_weight_log_train.csv"
        )

        weight_log_file_test = os.path.join(
            save_root,
            f"trainRounds_{train_rounds}_testRounds_{test_rounds}_num_jailbreaks_{num_experts}{f'_expert_set_{expert_set}' if expert_set is not None else ''}_weight_log_test.csv"
        )

        weight_df_train = pd.DataFrame(weight_log['train'], columns=[jb if jb is not None else "baseline" for jb in cfg.eval.jailbreaks])

        weight_df_test = pd.DataFrame(weight_log['test'], columns=[jb if jb is not None else "baseline" for jb in cfg.eval.jailbreaks])

        weight_df_train.to_csv(weight_log_file_train, index=False)
        weight_df_test.to_csv(weight_log_file_test, index=False)