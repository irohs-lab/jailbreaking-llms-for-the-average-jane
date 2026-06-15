import pandas as pd
import numpy as np
import config.main

def save_avg_asr(cfg:config.main.ProjectConfig, results:dict[dict[str,np.float64]], nan_instances:list[dict]):
    
    nan_key = "_NAN" if len(nan_instances)>0 else ""
    eval_split_key = f"_{cfg.metric_config.eval_split}" if cfg.metric_config.eval_split != 'test' else ''
    save_file_template = f"average_asr_over_domains_results{nan_key}{eval_split_key}.csv"
    splits = ("full", "simple", "complex")
    save_files = {split: save_file_template.removesuffix(".csv")+"_"+split+".csv" for split in splits}

    for split, save_path in save_files.items():
        pd.DataFrame(results[split]).to_csv(save_path)

def save_baseline_res(cfg:config.main.ProjectConfig, results:dict[dict[dict[str,np.float64]]]):

    simple_res_file = "baseline_asr_per_model_simple.csv"
    complex_res_file = "baseline_asr_per_model_complex.csv"
    overall_res_file = "baseline_asr_per_model.csv"

    if cfg.metric_config.eval_split != 'test':
        simple_res_file = simple_res_file.removesuffix('.csv') + "_" + cfg.metric_config.eval_split + ".csv"
        complex_res_file = complex_res_file.removesuffix('.csv') + "_" + cfg.metric_config.eval_split + ".csv"
        overall_res_file = overall_res_file.removesuffix('.csv') + "_" + cfg.metric_config.eval_split + ".csv"

    simple_df = []
    complex_df = []
    overall_df = []

    for model, model_res in results.items():
        simple_df.append({
            'model':model,
            'asr': model_res['overall']['simple']
        })

        complex_df.append({
            'model':model,
            'asr': model_res['overall']['complex']
        })
        
        overall_df.append({
            'model':model,
            'asr': model_res['overall']['complex']
        })
    
    pd.DataFrame(simple_df).to_csv(simple_res_file, index=False)
    pd.DataFrame(complex_df).to_csv(complex_res_file, index=False)
    pd.DataFrame(overall_df).to_csv(overall_res_file,index=False)