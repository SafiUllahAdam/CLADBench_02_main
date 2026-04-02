"""CSV export for cross-validation results."""

import csv
import json
import glob
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _stats_row(label: str, values) -> Dict[str, Any]:
    """Summary stats for a metric across trials."""
    v = np.array(values, dtype=float)
    return {
        "method": label, "mean": np.nanmean(v), "std": np.nanstd(v),
        "min": np.nanmin(v), "q25": np.nanpercentile(v, 25),
        "median": np.nanmedian(v), "q75": np.nanpercentile(v, 75),
        "max": np.nanmax(v), "n_trials": int(np.sum(~np.isnan(v))),
    }


def _next_test_id(results_dir: Path) -> int:
    """Auto-increment test ID from existing CSV filenames."""
    existing = [int(m.group(1)) for f in glob.glob(str(results_dir / "*.csv"))
                if (m := re.match(r"^(\d{3})\.", Path(f).name))]
    return max(existing, default=0) + 1


def build_filename(test_id: int, cv_results: Dict, colearner_name: str,
                   strategy_name: str, has_recurrent: bool) -> str:
    """Encode run metadata into the filename."""
    n_models = len(cv_results["model_names"])
    models_tag = "-".join(cv_results["model_names"])
    if has_recurrent:
        models_tag += "-gru"
    return f"{test_id:03d}.{n_models}.{int(has_recurrent)}.{colearner_name}.{strategy_name}.{models_tag}.csv"


def export_cv_results(cv_results: Dict, results_dir: Path,
                      colearner_name: str, strategy_name: str,
                      dataset_name: str, dataset_path: str,
                      train_config: Dict, has_recurrent: Optional[bool] = None,
                      project_root: Optional[Path] = None) -> Path:
    """Write stats CSV + raw per-trial CSV. Returns stats filepath."""
    results_dir.mkdir(parents=True, exist_ok=True)

    if has_recurrent is None:
        has_recurrent = bool(any(not np.isnan(a) for a in cv_results["gru"]["auc"]))

    test_id = _next_test_id(results_dir)
    filename = build_filename(test_id, cv_results, colearner_name, strategy_name, has_recurrent)
    filepath = results_dir / filename

    # Build summary rows
    rows = []
    for name in cv_results["model_names"]:
        rows.append(_stats_row(f"solo_{name}_auc", cv_results["solo"][name]["auc"]))
        rows.append(_stats_row(f"solo_{name}_ap", cv_results["solo"][name]["ap"]))
    for name in cv_results["model_names"]:
        rows.append(_stats_row(f"collab_{name}_auc", cv_results["collab"][name]["auc"]))
        rows.append(_stats_row(f"collab_{name}_ap", cv_results["collab"][name]["ap"]))
    rows.append(_stats_row("collab_ensemble_auc", cv_results["collab"]["ensemble"]["auc"]))
    rows.append(_stats_row("collab_ensemble_ap", cv_results["collab"]["ensemble"]["ap"]))
    if has_recurrent:
        rows.append(_stats_row("gru_judge_auc", cv_results["gru"]["auc"]))
        rows.append(_stats_row("gru_judge_ap", cv_results["gru"]["ap"]))
    for name in cv_results["model_names"]:
        da = np.array(cv_results["collab"][name]["auc"]) - np.array(cv_results["solo"][name]["auc"])
        dp = np.array(cv_results["collab"][name]["ap"]) - np.array(cv_results["solo"][name]["ap"])
        rows.append(_stats_row(f"delta_{name}_auc", da))
        rows.append(_stats_row(f"delta_{name}_ap", dp))

    # Run metadata (embedded as JSON header in CSV)
    metadata = {
        "test_id": test_id, "dataset": dataset_name, "dataset_path": dataset_path,
        "n_trials": cv_results["n_trials"], "seeds": cv_results["seeds"],
        "models": cv_results["model_names"],
        "n_models": len(cv_results["model_names"]),
        "has_recurrent": has_recurrent, "recurrent_model": "gru" if has_recurrent else "none",
        "colearner": colearner_name, "strategy": strategy_name,
        "warmup_epochs": train_config.get("warmup_epochs"),
        "max_chapters": train_config.get("max_chapters"),
        "epochs_per_chapter": train_config.get("epochs_per_chapter"),
        "gru_start_chapter": train_config.get("gru_start_chapter") if has_recurrent else None,
        "train_split": train_config.get("train_split"),
        "val_test_split": train_config.get("val_test_split"),
        "semi_label_ratio": train_config.get("semi_label_ratio"),
        "timestamp": datetime.now().isoformat(),
    }

    # Stats CSV
    stat_cols = ["method", "mean", "std", "min", "q25", "median", "q75", "max", "n_trials"]
    with open(filepath, "w", newline="") as f:
        f.write(f"#META:{json.dumps(metadata)}\n")
        writer = csv.DictWriter(f, fieldnames=stat_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]
                             for k in stat_cols})

    # Raw per-trial CSV
    raw_filepath = filepath.with_suffix(".raw.csv")
    raw_cols = ["trial", "seed", "method", "auc", "ap"]
    with open(raw_filepath, "w", newline="") as f:
        f.write(f"#META:{json.dumps(metadata)}\n")
        writer = csv.DictWriter(f, fieldnames=raw_cols)
        writer.writeheader()
        for t in range(cv_results["n_trials"]):
            seed = cv_results["seeds"][t]
            for name in cv_results["model_names"]:
                writer.writerow({"trial": t, "seed": seed, "method": f"solo_{name}",
                                 "auc": f"{cv_results['solo'][name]['auc'][t]:.6f}",
                                 "ap": f"{cv_results['solo'][name]['ap'][t]:.6f}"})
            for name in cv_results["model_names"]:
                writer.writerow({"trial": t, "seed": seed, "method": f"collab_{name}",
                                 "auc": f"{cv_results['collab'][name]['auc'][t]:.6f}",
                                 "ap": f"{cv_results['collab'][name]['ap'][t]:.6f}"})
            writer.writerow({"trial": t, "seed": seed, "method": "collab_ensemble",
                             "auc": f"{cv_results['collab']['ensemble']['auc'][t]:.6f}",
                             "ap": f"{cv_results['collab']['ensemble']['ap'][t]:.6f}"})
            if has_recurrent:
                g_auc = cv_results["gru"]["auc"][t]
                g_ap = cv_results["gru"]["ap"][t]
                writer.writerow({"trial": t, "seed": seed, "method": "gru_judge",
                                 "auc": "" if np.isnan(g_auc) else f"{g_auc:.6f}",
                                 "ap": "" if np.isnan(g_ap) else f"{g_ap:.6f}"})

    rel = filepath.relative_to(project_root) if project_root else filepath
    raw_rel = raw_filepath.relative_to(project_root) if project_root else raw_filepath
    logger.info(f"Stats  -> {rel}")
    logger.info(f"Raw    -> {raw_rel}")
    print(f"✓ Stats  → {rel}")
    print(f"✓ Raw    → {raw_rel}")
    return filepath
