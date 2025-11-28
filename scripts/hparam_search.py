from __future__ import annotations

import csv
import pytz
import json
import torch
import asyncio
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List

from src.graph_builder import GraphManager, WINDOW_SIZE, CORRELATION_THRESHOLD
from src.train import TrainingSummary, train_model


TRAIN_PARAM_KEYS = ["num_epoch", "learning_rate", "batch_size"]
MODEL_PARAM_KEYS = [
    "hidden_dim",
    "out_dim",
    "dropout",
    "embed_l2_reg",
    "normalize_embeddings",
]
GRAPH_PARAM_KEYS = ["window_size", "corr_threshold"]

DEFAULT_SEARCH_SPACE: Dict[str, Iterable[Any]] = {
    "window_size": [29, 36],  # n % 7 == 1 and n > 20
    "corr_threshold": [0.75, 0.6],
    "num_epoch": [90, 120],
    "hidden_dim": [64, 96],
    "out_dim": [32, 64],
    "dropout": [0.3, 0.4],
    "normalize_embeddings": [True, False],
}


def _load_space(path: str | None) -> Dict[str, Iterable[Any]]:
    if not path:
        return DEFAULT_SEARCH_SPACE
    override_path = Path(path)
    with override_path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if not isinstance(content, dict):
        raise ValueError("Search space JSON must be an object mapping keys to lists.")
    return {k: v for k, v in content.items()}


def _expand(search_space: Dict[str, Iterable[Any]]):
    keys = list(search_space.keys())
    values = [list(search_space[key]) for key in keys]
    for combo in itertools.product(*values):
        yield {key: value for key, value in zip(keys, combo)}


async def _run_single(
    GM: GraphManager,
    config: Dict[str, Any],
    output_dir: Path,
    run_id: int,
    notes: str,
) -> Dict[str, Any]:
    run_dir = output_dir / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {k: config[k] for k in MODEL_PARAM_KEYS if k in config}

    summary: TrainingSummary = await train_model(
        GM,
        num_epoch=config.get("num_epoch", 100),
        learning_rate=config.get("learning_rate", 1e-3),
        batch_size=config.get("batch_size", 128),
        model_kwargs=model_kwargs,
    )

    best_epoch = summary.best_epoch or {}
    final_epoch = summary.history[-1] if summary.history else {}
    timestamp = datetime.now(tz=pytz.timezone("Asia/Hong_Kong")).isoformat()

    history_path = run_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": timestamp,
                "config": summary.config,
                "history": summary.history,
                "notes": notes,
            },
            handle,
            indent=2,
        )

    torch.save(summary.model.state_dict(), run_dir / "model.pth")

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        **{key: config.get(key) for key in GRAPH_PARAM_KEYS + TRAIN_PARAM_KEYS},
        **{key: config.get(key) for key in MODEL_PARAM_KEYS},
        "best_epoch": best_epoch.get("epoch"),
        "best_loss": best_epoch.get("avg_loss"),
        "final_epoch": final_epoch.get("epoch"),
        "final_loss": final_epoch.get("avg_loss"),
        "graphs_last_epoch": final_epoch.get("graphs"),
        "batches_last_epoch": final_epoch.get("batches"),
        "notes": notes,
        "history_path": str(history_path),
    }


def _write_summary(rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.csv"
    if not rows:
        return summary_path

    fieldnames = list(rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


async def _runner(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = _load_space(args.search_space)
    combos = list(_expand(search_space))
    if args.max_combos:
        combos = combos[: args.max_combos]

    results: List[Dict[str, Any]] = []

    temp_GM = GraphManager()
    features = await temp_GM.async_gather_features(20)

    for run_id, combo in enumerate(combos, start=1):
        if run_id < args.start_from:
            continue
        
        GM = GraphManager(
            window_size=combo.get("window_size", WINDOW_SIZE),
            corr_threshold=combo.get("corr_threshold", CORRELATION_THRESHOLD),
        )
        GM.features = features
        print(f"Starting run {run_id}/{len(combos)} with config: {combo}")
        try:
            result = await _run_single(GM, combo, output_dir, run_id, args.notes)
            results.append(result)
            print(
                f"Finished run {run_id}: best_loss={result['best_loss']} "
                f"at epoch {result['best_epoch']}"
            )
        except Exception as exc:  # noqa: BLE001
            failure_path = output_dir / f"run_{run_id:03d}" / "error.txt"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(str(exc), encoding="utf-8")
            print(f"Run {run_id} failed, see {failure_path}")

    summary_path = _write_summary(results, output_dir)
    if results:
        top = sorted(
            [row for row in results if row.get("best_loss") is not None],
            key=lambda row: row["best_loss"],
        )[: args.top_k]
        print("\nTop performing configurations:")
        for rank, row in enumerate(top, start=1):
            param_keys = GRAPH_PARAM_KEYS + TRAIN_PARAM_KEYS + MODEL_PARAM_KEYS
            config_bits = [
                f"{key}={row.get(key)}"
                for key in param_keys
                if row.get(key) is not None
            ]
            print(
                f"{rank}. loss={row['best_loss']:.5f} "
                f"epoch={row['best_epoch']} config={{ {', '.join(config_bits)} }}"
            )
    print(f"\nSaved summary to {summary_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search TwoTowerSAGE hyperparameters and log results."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/hparam_runs",
        help="Directory for per-run logs and summary CSV.",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        help="Optional path to JSON file overriding the default search space.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="Limit the number of combinations evaluated (0 = all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top configurations to display at the end.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form string stored with each run for later reporting.",
    )

    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Run ID to start from, useful for resuming interrupted searches.",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_runner(args))


if __name__ == "__main__":
    main()
