"""Export presentation figures from actual prediction CSVs and training history."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from agentic_workflows.restaurant_evaluation import score_restaurant_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/restaurant")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/phi3-restaurant-adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    frame = pd.read_csv(args.data_dir / "evaluation_results.csv", keep_default_na=False)
    _, metrics = score_restaurant_predictions(frame)
    if "base_response" not in frame:
        parser.error("A paired base/adapted run is required.")
    splits = sorted(frame["source_split"].unique())
    if len(splits) != 1 or splits[0] not in {"validation", "test"}:
        parser.error("Plot one official validation or test split at a time.")
    split = splits[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    kinds = ("rouge1", "rouge2", "rougeL")
    xs = list(range(3))
    for offset, label, title, color in ((-0.18, "base", "Base Phi-3", "#64748b"), (0.18, "finetuned", "Phi-3 + LoRA", "#0d9488")):
        bars = axis.bar([x + offset for x in xs], [metrics[f"{label}_{kind}"] for kind in kinds], width=0.34, label=title, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    axis.set(xticks=xs, xticklabels=["ROUGE-1", "ROUGE-2", "ROUGE-L"], ylim=(0, 1), ylabel="F1 score",
             title=f"Restaurant descriptions — {split}, {len(frame)} unique inputs")
    axis.legend(frameon=False, loc="upper left")
    axis.yaxis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    fig.savefig(args.output_dir / f"restaurant_{split}_rouge.png", dpi=180)
    fig.savefig(args.output_dir / f"restaurant_{split}_rouge.pdf")
    plt.close(fig)
    history_path = args.model_dir / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        training = [row for row in history if "loss" in row]
        validation = [row for row in history if "eval_loss" in row]
        fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
        axis.plot([row["step"] for row in training], [row["loss"] for row in training], color="#0d9488", label="Logged training loss")
        axis.scatter([row["step"] for row in validation], [row["eval_loss"] for row in validation], color="#be123c", marker="D", label="Official validation loss", zorder=3)
        axis.set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="Phi-3 QLoRA training")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)
        fig.savefig(args.output_dir / "restaurant_training.png", dpi=180)
        fig.savefig(args.output_dir / "restaurant_training.pdf")
        plt.close(fig)
    print(f"Figures exported to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
