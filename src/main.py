"""CoBench CLI — collaborative anomaly detection benchmark"""
import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_config import PROJECT_ROOT, ADBENCH_ROOT, EVAL_DATASETS, RESULTS_DIR, CV_DEFAULTS
if str(ADBENCH_ROOT) not in sys.path:
    sys.path.append(str(ADBENCH_ROOT))

from baselines.adbench.ModelWrapperADBench import get_model_detector_dict
from colearner import CoLearnerVal
from runner import STRATEGY_REGISTRY, run_cv, print_summary, generate_plots
from results_io import export_cv_results
from utils import set_seed

logger = logging.getLogger("cobench")
_D = CV_DEFAULTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--recurrent_judge", default=None, choices=["gru", None])
    p.add_argument("--n_trials", type=int, default=_D["n_trials"])
    p.add_argument("--data_to_test", nargs="+", default=["annthyroid"])
    p.add_argument("--save_results", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validation_loss_plots", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--available_ratio_of_data", type=float, default=_D["semi_label_ratio"])
    p.add_argument("--colearning_strategy", default="plateau", choices=list(STRATEGY_REGISTRY.keys()))
    p.add_argument("--seed", type=int, default=_D["seed"])
    p.add_argument("--warmup_epochs", type=int, default=_D["warmup_epochs"])
    p.add_argument("--max_chapters", type=int, default=_D["max_chapters"])
    p.add_argument("--epochs_per_chapter", type=int, default=_D["epochs_per_chapter"])
    p.add_argument("--gru_start_chapter", type=int, default=_D["gru_start_chapter"])
    p.add_argument("--train_split", type=float, default=_D["train_split"])
    p.add_argument("--val_test_split", type=float, default=_D["val_test_split"])
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    available_models = list(get_model_detector_dict().keys())
    for ds in args.data_to_test:
        if ds not in EVAL_DATASETS:
            sys.exit(f"Unknown dataset '{ds}'. Available: {list(EVAL_DATASETS.keys())}")
    for m in args.models:
        if m not in available_models:
            sys.exit(f"Unknown model '{m}'. Available: {available_models}")

    set_seed(args.seed)
    has_gru = args.recurrent_judge == "gru"
    train_config = {k: getattr(args, k) for k in
                    ("warmup_epochs", "max_chapters", "epochs_per_chapter",
                     "gru_start_chapter", "train_split", "val_test_split")}
    train_config["semi_label_ratio"] = args.available_ratio_of_data

    t0 = time.time()
    print(f"\nCoBench — {len(args.data_to_test)} dataset(s) × {args.n_trials} trials")
    print(f"  Models: {args.models} | GRU: {has_gru} | Strategy: {args.colearning_strategy}")
    print(f"  Labeled ratio: {args.available_ratio_of_data} | Chapters: {args.max_chapters}×{args.epochs_per_chapter}ep\n")

    for dataset_name in args.data_to_test:
        dataset_path = EVAL_DATASETS[dataset_name]
        print(f"\n{'═'*72}\n  Dataset: {dataset_name}  ({dataset_path.name})\n{'═'*72}")
        cv_results = run_cv(args, dataset_name, dataset_path)
        print_summary(cv_results, dataset_name, has_gru)
        if args.save_results:
            fp = export_cv_results(
                cv_results, RESULTS_DIR, CoLearnerVal.__name__,
                STRATEGY_REGISTRY[args.colearning_strategy].__name__,
                dataset_name, str(dataset_path), train_config,
                has_recurrent=has_gru, project_root=PROJECT_ROOT)
            if args.validation_loss_plots:
                generate_plots(cv_results, dataset_name, has_gru, fp)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
