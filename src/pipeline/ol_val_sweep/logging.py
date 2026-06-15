import os
import json
import pandas as pd
from collections import defaultdict

from rich.console import Console
from rich.table import Table


console = Console()


def select_best_params(cfg, sweep_results):

    grouped = defaultdict(lambda: {"scores": [], "model_scores": {}})

    for r in sweep_results:

        key = tuple(sorted(r["params"].items()))
        grouped[key]["scores"].append(r["score"])
        grouped[key]["model_scores"][r["target_model"]] = r["score"]

    aggregated = []

    for key, entry in grouped.items():

        params = dict(key)
        scores = entry["scores"]

        aggregated.append(
            dict(
                params=params,
                mean_asr=sum(scores) / len(scores),
                model_scores=entry["model_scores"]
            )
        )

    aggregated.sort(key=lambda x: x["mean_asr"], reverse=True)

    table = Table(title="Top 5 Hyperparameter Settings (Mean ASR across models)")

    table.add_column("Rank", justify="right")
    table.add_column("Mean ASR", justify="right")
    table.add_column("Params")

    for i, entry in enumerate(aggregated[:5], start=1):

        table.add_row(
            str(i),
            f"{entry['mean_asr']:.4f}",
            str(entry["params"])
        )

    console.print("\n")
    console.print(table)

    best = aggregated[0]

    return dict(
        best_params=best["params"],
        aggregated_results=aggregated
    )


def write_best_config(cfg, aggregated_results):

    output_dir = os.path.join(
        cfg.paths.result_dir,
        "ol_val_sweep_results"
    )

    os.makedirs(output_dir, exist_ok=True)

    algo = cfg.ol_scheme.name

    json_path = os.path.join(output_dir, f"{algo}_sweep_results.json")
    csv_path = os.path.join(output_dir, f"{algo}_sweep_results.csv")

    with open(json_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    rows = []

    for entry in aggregated_results:

        row = dict(entry["params"])
        row["mean_asr"] = entry["mean_asr"]

        for model, score in entry["model_scores"].items():
            row[f"asr_{model.split('/')[-1]}"] = score

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)

    console.print(f"\n[green]Sweep results written to:[/green]")
    console.print(f"JSON: {json_path}")
    console.print(f"CSV : {csv_path}")