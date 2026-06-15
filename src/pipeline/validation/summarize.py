import config.main
from config.model import get_model_config_name

import pandas as pd
import os
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from ast import literal_eval
import json
from itertools import combinations
import textwrap
from tqdm import tqdm

def _save_cm_heatmap(cm, labels, path, title, model=None, prompt_id=None, combo_name=None):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums, where=row_sums != 0)

    pink_cmap = LinearSegmentedColormap.from_list(
        "modern_pink",
        ["#fff1f5", "#f7b2c4", "#e75480", "#c2185b"]
    )

    fig, ax = plt.subplots(figsize=(4, 5))

    im = ax.imshow(cm, cmap=pink_cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title("Confusion Matrix", fontsize=12, pad=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.6 else "black",
                fontsize=10,
                fontweight="medium"
            )

    meta_lines = []
    if model:
        meta_lines.append(f"Model: {model}")
    if prompt_id:
        wrapped_prompt = "\n".join(textwrap.wrap(prompt_id, width=45))
        meta_lines.append(f"Prompt: {wrapped_prompt}")
    if combo_name:
        wrapped_combo = "\n".join(textwrap.wrap(combo_name, width=45))
        meta_lines.append(f"Combo: {wrapped_combo}")

    if meta_lines:
        fig.text(
            0.5, 0.92,
            "\n".join(meta_lines),
            ha="center",
            va="top",
            fontsize=9
        )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _confusion_matrix(y_true, y_pred, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            cm[i, j] = int(
                ((y_true == true_label) & (y_pred == pred_label)).sum()
            )
    return cm


def fetch_validation_results(cfg: config.main.ProjectConfig):

    judge_type = cfg.llm_judge.judge_type
    save_dir = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        cfg.paths.val_responses_dir,
        judge_type
    )

    results = []

    for fname in os.listdir(save_dir):
        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(save_dir, fname))

        results.append({
            "model": df["judge_model"].iloc[0],
            "backend": df["judge_backend"].iloc[0],
            "prompt_id": df["prompt_id"].iloc[0],
            "output_mode": df["output_mode"].iloc[0],
            "label_space": literal_eval(df["label_space"].iloc[0]),
            "df": df
        })

    return dict(result_dfs=results)


def compute_diagnostics_rater(cfg:config.main.ProjectConfig, result_dfs: list[dict]):
    out_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        "validation_diagnostics",
        cfg.llm_judge.judge_type
    )

    os.makedirs(out_root, exist_ok=True)

    summary_rows = []

    for entry in result_dfs:
        model = entry["model"]
        prompt_id = entry["prompt_id"]
        output_mode = entry["output_mode"]
        labels = entry["label_space"]
        df = entry["df"]
        total = len(df)
        invalid = df["judge_rating"].isna().sum()
        valid_rate = 1 - invalid / total

        out_dir = os.path.join(out_root, model)
        os.makedirs(out_dir, exist_ok=True)

        y = df["rating"].astype(float)
        yhat = df["judge_rating"].astype(float)

        label_names = {
            0.0: "Harmful",
            0.5: "Vague",
            1.0: "Benign"
        }
        display_labels = [label_names[l] for l in labels]

        cm = _confusion_matrix(y, yhat, classes=labels)
        cm_npy_path = os.path.join(out_dir, f"cm_{prompt_id}.npy")
        np.save(cm_npy_path, cm)
        _save_cm_heatmap(
            cm,
            display_labels,
            os.path.join(out_dir, f"cm_{prompt_id}.png"),
            title="",
            model=model,
            prompt_id=prompt_id
        )

        metrics = {
            "accuracy": (y == yhat).mean(),
        }

        with open(os.path.join(out_dir, f"metrics_{prompt_id}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        summary_rows.append({
            "model": model,
            "prompt_id": prompt_id,
            "output_mode": output_mode,
            "accuracy": metrics["accuracy"],
            "format_valid_rate": valid_rate,
            "invalid_count": invalid,
            "use_icl": df["use_icl"].iloc[0] if "use_icl" in df.columns else True
        })

    return dict(summary_df=pd.DataFrame(summary_rows))

def compute_diagnostics_classifier(cfg:config.main.ProjectConfig, result_dfs:list[dict]):
    
    sweep_root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.cache_dir,
        "validation_sweeps",
        "classifier"
    )

    full_sweep_path = os.path.join(sweep_root, "full_sweep.csv")

    summary_rows = []
    full_rows = []

    pbar = tqdm(result_dfs)
    for entry in pbar:
        pbar.set_description(f"Model: {entry['model']}")
        model = entry["model"]
        prompt_id = entry["prompt_id"]
        output_mode = entry["output_mode"]
        df = entry["df"]
        total = len(df)
        invalid = df["judge_label"].isna().sum()
        valid_rate = 1 - invalid / total

        sweep_path = os.path.join(
            sweep_root,
            model,
            f"{prompt_id}__{output_mode}.parquet"
        )

        sweep_df = pd.read_parquet(sweep_path)

        max_acc = sweep_df["acc"].max()
        best_rows = sweep_df[np.isclose(sweep_df["acc"], max_acc)]
        for _,best_row in best_rows.iterrows():
            summary_rows.append({
                "model": model,
                "prompt_id": prompt_id,
                "output_mode": output_mode,
                "best_setting": best_row["setting_key"],
                "best_accuracy": best_row["acc"],
                "format_valid_rate": valid_rate,
                "invalid_count": invalid
            })
        if not os.path.exists(full_sweep_path):
            for _,row in tqdm(sweep_df.iterrows(), desc="Iterating over rows"):
                full_rows.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "output_mode": output_mode,
                    "setting": row["setting_key"],
                    "accuracy": row["acc"],
                    "format_valid_rate": valid_rate,
                    "invalid_count": invalid
                })
    
    if not os.path.exists(full_sweep_path):
        pd.DataFrame(full_rows).to_csv(full_sweep_path, index=False)

    return dict(summary_df=pd.DataFrame(summary_rows))
    
     
def display_summary_rater(cfg: config.main.ProjectConfig, summary_df: pd.DataFrame):
    console = Console()

    if summary_df.empty:
        console.print("[bold red]No validation results found.[/bold red]")
        return

    summary_df = summary_df.sort_values(
        by=["output_mode", "accuracy"],
        ascending=[True, False]
    )

    # Split by ICL usage
    icl_df = summary_df[summary_df["use_icl"] == True]
    noicl_df = summary_df[summary_df["use_icl"] == False]

    def render_table(df, title_suffix):
        if df.empty:
            return

        for output_mode, group_df in df.groupby("output_mode"):
            table = Table(
                title=f"\nRater Validation Summary "
                      f"[bold cyan]({output_mode})[/bold cyan] "
                      f"[bold yellow]{title_suffix}[/bold yellow]"
            )

            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Prompt ID", style="magenta", no_wrap=True)
            table.add_column("Accuracy", justify="right")
            table.add_column("Valid %", justify="right")
            table.add_column("Invalid", justify="right")

            for _, r in group_df.iterrows():
                table.add_row(
                    r["model"],
                    r["prompt_id"],
                    f"{r['accuracy'] * 100:.2f}%",
                    f"{r['format_valid_rate'] * 100:.1f}%",
                    str(r["invalid_count"])
                )

            console.print(table)

    # First show ICL results
    render_table(icl_df, "With ICL")

    # Then show Zero-shot results
    render_table(noicl_df, "Without ICL")

def display_summary_classifier(
    cfg: config.main.ProjectConfig,
    summary_df: pd.DataFrame
):

    from rich.console import Console
    from rich.table import Table

    console = Console()

    if summary_df.empty:
        console.print("[bold red]No classifier validation results found.[/bold red]")
        return

    for output_mode, group_df in summary_df.groupby("output_mode"):

        group_df = group_df.sort_values(
            "best_accuracy",
            ascending=False
        )
        group_df = group_df.head(50)

        table = Table(
            title=f"\nClassifier Validation Summary "
                  f"[bold cyan]({output_mode})[/bold cyan]"
        )

        table.add_column("Model", style="cyan")
        table.add_column("Prompt ID", style="magenta")
        table.add_column("Best Setting", style="yellow")
        table.add_column("Accuracy", justify="right")
        table.add_column("Valid %", justify="right")
        table.add_column("Invalid", justify="right")

        for _, row in group_df.iterrows():
            table.add_row(
                row["model"],
                row["prompt_id"],
                row["best_setting"],
                f"{row['best_accuracy'] * 100:.2f}%",
                f"{row['format_valid_rate'] * 100:.1f}%",
                str(row["invalid_count"])
            )

        console.print(table)


def compute_diagnostics(cfg:config.main.ProjectConfig):
    if cfg.llm_judge.judge_type == 'rater':
        func = compute_diagnostics_rater
    elif cfg.llm_judge.judge_type == 'classifier':
        func = compute_diagnostics_classifier
    else:
        raise NotImplementedError(f"compute_diagnostics() not implemented for judge type: {cfg.llm_judge.judge_type}")
    
    func.__name__ = "compute_diagnostics"
    return func

def display_summary(cfg:config.main.ProjectConfig):
    if cfg.llm_judge.judge_type == 'rater':
        func = display_summary_rater
    elif cfg.llm_judge.judge_type == 'classifier':
        func = display_summary_classifier
    else:
        raise NotImplementedError(f"display_summary() not implemented for judge type: {cfg.llm_judge.judge_type}")
    
    func.__name__ = "display_summary"
    return func

    








    