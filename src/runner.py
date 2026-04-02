"""CV runner: solo baselines, collaborative training, optional GRU judge"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from benchmark_config import CV_DEFAULTS, PROJECT_ROOT
from baselines.adbench.ModelWrapperADBench import get_model_detector_dict
from baselines.adbench.data_loader import ClassicalADBenchData
from colearner import CoLearnerVal, DelayedRecurrentCoLearner
from recurrentmodels import GRURecurrentModel
from strategy import PlateauStrategy, AdaptivePlateauStrategy, RecurrentPlateauStrategy
from utils import create_models, set_seed

logger = logging.getLogger("cobench.runner")

STRATEGY_REGISTRY = {
    "plateau": PlateauStrategy,
    "adaptive": AdaptivePlateauStrategy,
}


def _empty_gru_entry() -> dict:
    return {"auc": np.nan, "ap": np.nan, "train_loss": [], "val_loss": [], "chapters": {"chapters": []}}


def _score_model(model, y_test) -> dict:
    sc = model.predict_scores()
    return {
        "auc": roc_auc_score(y_test, sc),
        "ap": average_precision_score(y_test, sc),
        "scores": sc,
        "train_loss": getattr(model, "_train_loss_history", []).copy(),
        "val_loss": getattr(model, "_val_loss_history", []).copy(),
    }


def _append_metrics(store: dict, metrics: dict) -> None:
    for k in ("auc", "ap", "train_loss", "val_loss"):
        store[k].append(metrics[k])


def run_cv(args, dataset_name: str, dataset_path: Path) -> dict:
    registry = get_model_detector_dict()
    model_names = args.models
    has_gru = args.recurrent_judge == "gru"

    cv_results = {
        "n_trials": args.n_trials, "dataset": str(dataset_path),
        "model_names": list(model_names), "seeds": [],
        "solo": {n: {"auc": [], "ap": [], "train_loss": [], "val_loss": []} for n in model_names},
        "collab": {n: {"auc": [], "ap": [], "train_loss": [], "val_loss": []}
                   for n in list(model_names) + ["ensemble"]},
        "collab_chapters": [], "collab_val_loss": [],
        "gru": {"auc": [], "ap": [], "train_loss": [], "val_loss": [], "chapters": []},
    }

    for trial_idx in range(args.n_trials):
        seed = args.seed + trial_idx
        set_seed(seed)
        cv_results["seeds"].append(seed)
        t0 = time.time()
        logger.info(f"Trial {trial_idx + 1}/{args.n_trials} (seed={seed})")

        trial_data = ClassicalADBenchData(
            dataset_path, train_test_split_ratio=args.train_split,
            val_test_split_ratio=args.val_test_split,
            preserve_labeled=CV_DEFAULTS["preserve_labeled"],
            labeled_ratio=args.available_ratio_of_data,
            stratified=CV_DEFAULTS["semi_stratified"], random_state=seed,
        )

        # Solo baselines
        for name in model_names:
            solo_m = create_models(registry, [name], trial_data)[0]
            solo_m.fit()
            m_res = _score_model(solo_m, trial_data.y_test)
            _append_metrics(cv_results["solo"][name], m_res)
            logger.info(f"  Solo  {name:<10} AUC={m_res['auc']:.4f}  AP={m_res['ap']:.4f}")

        # Collaborative
        StrategyClass = STRATEGY_REGISTRY[args.colearning_strategy]
        collab_models = create_models(registry, model_names, trial_data)
        collab_strat = StrategyClass(patience=args.patience, mode="max", min_delta=0.001)
        collab_cl = CoLearnerVal(
            models=collab_models, data=trial_data, strategy=collab_strat,
            warmup_epochs=args.warmup_epochs, max_chapters=args.max_chapters,
            epochs_per_chapter=args.epochs_per_chapter,
        )
        collab_hist = collab_cl.cotrain(eval_interval=1)
        cv_results["collab_chapters"].append(collab_hist)
        cv_results["collab_val_loss"].append(
            {k: v.copy() for k, v in collab_cl.val_loss_history.items()})

        for name, m in zip(model_names, collab_models):
            m_res = _score_model(m, trial_data.y_test)
            _append_metrics(cv_results["collab"][name], m_res)
            logger.info(f"  Collab {name:<10} AUC={m_res['auc']:.4f}  AP={m_res['ap']:.4f}")

        ens_sc = np.mean([m.predict_scores() for m in collab_models], axis=0)
        ens_auc = roc_auc_score(trial_data.y_test, ens_sc)
        ens_ap = average_precision_score(trial_data.y_test, ens_sc)
        cv_results["collab"]["ensemble"]["auc"].append(ens_auc)
        cv_results["collab"]["ensemble"]["ap"].append(ens_ap)
        logger.info(f"  Collab ensemble   AUC={ens_auc:.4f}  AP={ens_ap:.4f}")

        # GRU judge (optional)
        if has_gru:
            _run_gru_trial(args, registry, model_names, trial_data, seed, cv_results)
        else:
            g = _empty_gru_entry()
            for k in ("auc", "ap", "train_loss", "val_loss"):
                cv_results["gru"][k].append(g[k])
            cv_results["gru"]["chapters"].append(g["chapters"])

        logger.info(f"  Trial time: {time.time() - t0:.1f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cv_results


def _run_gru_trial(args, registry, model_names, trial_data, seed, cv_results) -> None:
    try:
        gru_models = create_models(registry, model_names, trial_data)
        gru_m = GRURecurrentModel(
            hidden_size=128, num_layers=1, dropout=0.0, lr=1e-3,
            batch_size=256, num_epochs=5, seed=seed, n_detectors=len(gru_models),
        )
        gru_strat = RecurrentPlateauStrategy(
            recurrent_start_chapter=args.gru_start_chapter,
            fallback_key="ensemble", recurrent_key="gru", mode="max",
            patience_fallback=args.patience, patience_recurrent=3, min_delta=0.001,
        )
        gru_cl = DelayedRecurrentCoLearner(
            models=gru_models, data=trial_data, strategy=gru_strat,
            recurrent_model=gru_m, warmup_epochs=args.warmup_epochs,
            max_chapters=args.max_chapters, embeddings_aggregate="stack",
            recurrent_start_chapter=args.gru_start_chapter,
        )
        hist_gru = gru_cl.cotrain(eval_interval=1)

        gru_cl.embeddings_use_train = False
        test_emb = gru_cl._collect_embeddings(indexes=None)
        gru_sc = gru_m.predict_scores(test_emb)
        g_auc = roc_auc_score(trial_data.y_test, gru_sc)
        g_ap = average_precision_score(trial_data.y_test, gru_sc)
        cv_results["gru"]["auc"].append(g_auc)
        cv_results["gru"]["ap"].append(g_ap)
        cv_results["gru"]["train_loss"].append(getattr(gru_m, "_train_loss_history", []).copy())
        cv_results["gru"]["val_loss"].append(getattr(gru_m, "_val_loss_history", []).copy())
        cv_results["gru"]["chapters"].append(hist_gru)
        logger.info(f"  GRU judge      AUC={g_auc:.4f}  AP={g_ap:.4f}")
    except Exception as exc:
        logger.warning(f"  GRU FAILED: {exc}")
        g = _empty_gru_entry()
        for k in ("auc", "ap", "train_loss", "val_loss"):
            cv_results["gru"][k].append(g[k])
        cv_results["gru"]["chapters"].append(g["chapters"])


def _fmt(arr):
    return f"{np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}"


def print_summary(cv_results: dict, dataset_name: str, has_gru: bool) -> None:
    names = cv_results["model_names"]
    nt = cv_results["n_trials"]

    print(f"\n{'='*72}")
    print(f"  CROSS-VALIDATION SUMMARY  ({nt} trials · {dataset_name})")
    print(f"{'='*72}")
    print(f"  {'Method':<26} {'AUC (mean±std)':<22} {'AP (mean±std)':<22}")
    print(f"  {'-'*68}")

    for n in names:
        print(f"  Solo {n:<20} {_fmt(cv_results['solo'][n]['auc'])}     {_fmt(cv_results['solo'][n]['ap'])}")
    print(f"  {'-'*68}")
    for n in list(names) + ["ensemble"]:
        print(f"  Collab {n:<18} {_fmt(cv_results['collab'][n]['auc'])}     {_fmt(cv_results['collab'][n]['ap'])}")
    print(f"  {'-'*68}")

    if has_gru:
        print(f"  {'GRU recurrent':<26} {_fmt(cv_results['gru']['auc'])}     {_fmt(cv_results['gru']['ap'])}")

    print(f"\n  {'Δ (collab − solo)':<26} {'ΔAUC':<22} {'ΔAP':<22}")
    print(f"  {'-'*68}")
    for n in names:
        da = np.array(cv_results["collab"][n]["auc"]) - np.array(cv_results["solo"][n]["auc"])
        dp = np.array(cv_results["collab"][n]["ap"]) - np.array(cv_results["solo"][n]["ap"])
        print(f"  {n:<26} {np.mean(da):+.4f} ± {np.std(da):.4f}     {np.mean(dp):+.4f} ± {np.std(dp):.4f}")

    if has_gru:
        ga = np.array(cv_results["gru"]["auc"])
        if not np.all(np.isnan(ga)):
            d_auc = ga - np.array(cv_results["collab"]["ensemble"]["auc"])
            d_ap = np.array(cv_results["gru"]["ap"]) - np.array(cv_results["collab"]["ensemble"]["ap"])
            print(f"\n  {'Δ (GRU − ensemble)':<26} {'ΔAUC':<22} {'ΔAP':<22}")
            print(f"  {'-'*68}")
            print(f"  {'GRU judge':<26} {_fmt(d_auc)}     {_fmt(d_ap)}")


def generate_plots(cv_results: dict, dataset_name: str,
                   has_gru: bool, filepath: Path) -> None:
    from plotting import (COLORS, plot_chapter_dynamics, plot_final_bars,
                          plot_loss_evolution, save_figures)
    import matplotlib
    matplotlib.use("Agg")

    figures = {
        "fig_dynamics": plot_chapter_dynamics(cv_results, COLORS, dataset_name),
        "fig_barplot": plot_final_bars(cv_results, COLORS, dataset_name),
        "fig_losses": plot_loss_evolution(cv_results, COLORS, dataset_name, has_recurrent=has_gru),
    }
    save_figures(figures, filepath, PROJECT_ROOT)
