"""Publication-quality plots for CoBench CV results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from collections import OrderedDict

# Default palette
COLORS = {
    "solo": "#2E86AB", "collab": "#A23B72", "gru": "#F18F01",
    "prenet": "#06A77D", "deepsad": "#E63946", "ensemble": "#8338EC",
}

PUB_RC = {
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linewidth": 0.5,
}


def apply_pub_style():
    """Apply publication rc params."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(PUB_RC)


# ─── Internal helpers ────────────────────────────────────────────────

def _pad_and_stack(histories):
    """Stack variable-length lists into NaN-padded (n_trials × max_len) array."""
    if not histories or all(len(h) == 0 for h in histories):
        return None
    max_len = max(len(h) for h in histories)
    out = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        out[i, :len(h)] = h
    return out


def _lighten(color, amount=0.35):
    """Lighten a color by mixing with white."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


def _chapter_arr(chapter_histories, key):
    """Build (n_trials × max_chapters) NaN-padded array for a chapter metric key."""
    lengths = [len(h.get("chapters", [])) for h in chapter_histories]
    if not lengths or max(lengths) == 0:
        return None
    max_len = max(lengths)
    arr = np.full((len(chapter_histories), max_len), np.nan)
    for t, h in enumerate(chapter_histories):
        for c, chap in enumerate(h.get("chapters", [])):
            arr[t, c] = chap.get(key, np.nan)
    return arr


def _panel_label(ax, idx):
    """Add (a), (b), … panel annotation."""
    ax.text(-0.08, 1.06, f"({chr(ord('a') + idx)})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")


def _annotation(ax, text, pos=(0.03, 0.97), va="top", ha="left"):
    """Compact monospace annotation box."""
    ax.text(*pos, text, transform=ax.transAxes, fontsize=5.5, va=va, ha=ha,
            family="monospace", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", edgecolor="#cccccc", alpha=0.85, linewidth=0.5))


def _ribbon(ax, arr, color, label, marker="o", ls="-",
            stat="iqr", n_total=None, sparse_fade=True):
    """Mean ± band ribbon with individual trial traces.

    stat: 'iqr' (Q25-Q75) or 'std' (mean ± std).
    sparse_fade: if True, dotted line where majority of trials ended.
    """
    if arr is None or arr.size == 0:
        return False

    x = np.arange(1, arr.shape[1] + 1)
    mean = np.nanmean(arr, axis=0)
    if stat == "iqr":
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
    else:
        std = np.nanstd(arr, axis=0)
        lo, hi = mean - std, mean + std

    n_valid = np.sum(~np.isnan(arr), axis=0)
    n_tot = n_total or arr.shape[0]
    threshold = max(2, n_tot // 2)

    # Individual trial traces
    for t in range(arr.shape[0]):
        v = ~np.isnan(arr[t])
        ax.plot(x[v], arr[t][v], color=color, alpha=0.12, linewidth=0.6)

    if not sparse_fade:
        v = ~np.isnan(mean)
        ax.plot(x[v], mean[v], marker=marker, linewidth=1.8, markersize=3,
                color=color, ls=ls, label=label, alpha=0.9, zorder=3)
        ax.fill_between(x[v], lo[v], hi[v], color=color, alpha=0.18)
        return True

    # Reliable region (enough trials still running)
    reliable = n_valid >= threshold
    sparse = (~reliable) & (n_valid > 0)

    if np.any(reliable):
        idx = np.where(reliable)[0]
        end = min(idx[-1] + 2, len(x))
        sl = slice(0, end)
        ax.plot(x[sl], mean[sl], marker=marker, linewidth=1.8, markersize=3,
                color=color, ls=ls, label=label, alpha=0.9, zorder=3)
        ax.fill_between(x[sl], lo[sl], hi[sl], color=color, alpha=0.18)

    if np.any(sparse):
        first = np.where(sparse)[0][0]
        sl = slice(max(0, first - 1), None)
        v = n_valid[sl] > 0
        ax.plot(x[sl][v], mean[sl][v], marker=marker, linewidth=1.1,
                markersize=2.5, color=color, ls=":", alpha=0.4, zorder=3)
        ax.fill_between(x[sl][v], lo[sl][v], hi[sl][v], color=color, alpha=0.06)
        ax.axvline(x=x[first], color="gray", ls=":", linewidth=0.6, alpha=0.4)

    return True


# ─── Public API ──────────────────────────────────────────────────────

def plot_loss_ribbons(cv_results, colors=None, dataset_name=""):
    """Loss dynamics: mean ± std ribbon per model × {solo, collab, GRU judge}."""
    apply_pub_style()
    colors = colors or COLORS
    nt = cv_results["n_trials"]

    # Build configs grid: rows = model, cols = experiment
    configs = OrderedDict()
    for name in cv_results["model_names"]:
        configs[name] = OrderedDict(
            solo={"train": cv_results["solo"][name]["train_loss"],
                  "val":   cv_results["solo"][name]["val_loss"]},
            collab={"train": cv_results["collab"][name]["train_loss"],
                    "val":   cv_results["collab"][name]["val_loss"]},
        )
    configs["gru"] = OrderedDict(
        **{"GRU judge": {"train": cv_results["gru"]["train_loss"],
                         "val":   cv_results["gru"]["val_loss"]}})

    row_names = list(configs.keys())
    col_set = list(dict.fromkeys(c for rd in configs.values() for c in rd))
    nrows, ncols = len(row_names), len(col_set)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.8 * ncols, 2.8 * nrows), squeeze=False)
    for r, model_name in enumerate(row_names):
        color = colors.get(model_name, "#555")
        for c, cfg_name in enumerate(col_set):
            ax = axes[r][c]
            entry = configs[model_name].get(cfg_name)
            plotted = False
            if entry:
                for lt in ("train", "val"):
                    lc = _lighten(color, 0.35) if lt == "val" else color
                    ok = _ribbon(ax, _pad_and_stack(entry[lt]), lc, lt,
                                 "o" if lt == "train" else "s",
                                 "-" if lt == "train" else "--",
                                 stat="std", n_total=nt)
                    plotted = plotted or ok
            if plotted:
                ax.legend(loc="best", frameon=True, framealpha=0.9,
                          edgecolor="#cccccc")
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="#cccccc")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            if c == 0:
                ax.set_ylabel(model_name, fontweight="bold")
            if r == 0:
                ax.set_title(cfg_name, fontweight="bold")
            _panel_label(ax, r * ncols + c)

    fig.suptitle(
        f"Training loss dynamics (mean ± s.d., {nt} trials) — {dataset_name}",
        fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def plot_chapter_dynamics(cv_results, colors=None, dataset_name=""):
    """(a) Collab AUC trajectories + solo baselines, (b) ΔAUC per chapter."""
    apply_pub_style()
    colors = colors or COLORS
    nt = cv_results["n_trials"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) Collab AUC + solo baselines
    ax = axes[0]
    for name in cv_results["model_names"]:
        solo_mean = np.mean(cv_results["solo"][name]["auc"])
        ax.axhline(y=solo_mean, color=colors.get(name, "#555"), ls="--",
                   linewidth=1.0, alpha=0.5,
                   label=f"solo {name} = {solo_mean:.3f}")

    for i, name in enumerate(cv_results["model_names"]):
        arr = _chapter_arr(cv_results["collab_chapters"], f"model_{i}")
        _ribbon(ax, arr, colors.get(name, "#555"),
                f"collab {name}", "o", n_total=nt)

    arr_ens = _chapter_arr(cv_results["collab_chapters"], "ensemble")
    _ribbon(ax, arr_ens, colors["ensemble"], "collab ensemble", "D", n_total=nt)

    arr_gru = _chapter_arr(cv_results["gru"]["chapters"], "gru")
    if arr_gru is not None and not np.all(np.isnan(arr_gru)):
        _ribbon(ax, arr_gru, colors["gru"], "GRU judge", "s", n_total=nt)

    ax.set_title("Collaborative AUC over co-training chapters", fontweight="bold")
    ax.set_xlabel("chapter"); ax.set_ylabel("validation AUC"); ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fontsize=6)
    _panel_label(ax, 0)
    _annotation(ax,
                "solid = mean | shaded = IQR (Q25–Q75)\n"
                "thin = individual trials\n"
                "dashed horiz. = solo baseline")

    # (b) ΔAUC = collab − solo
    ax = axes[1]
    ax.axhline(y=0, color="#999", linewidth=0.8, zorder=1)

    for i, name in enumerate(cv_results["model_names"]):
        arr = _chapter_arr(cv_results["collab_chapters"], f"model_{i}")
        if arr is None:
            continue
        solo_aucs = np.array(cv_results["solo"][name]["auc"])
        _ribbon(ax, arr - solo_aucs[:, None],
                colors.get(name, "#555"), name, "o", n_total=nt)

    if arr_ens is not None:
        solo_avg = np.mean(
            [cv_results["solo"][n]["auc"]
             for n in cv_results["model_names"]], axis=0)
        _ribbon(ax, arr_ens - solo_avg[:, None],
                colors["ensemble"], "ensemble", "D", n_total=nt)

    ax.set_title("Collaboration gain (ΔAUC = collab − solo)", fontweight="bold")
    ax.set_xlabel("chapter"); ax.set_ylabel("ΔAUC (collab − solo)")
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    _panel_label(ax, 1)
    _annotation(ax,
                "above 0 = collab improves over solo\n"
                "shaded = IQR across trials")

    fig.suptitle(
        f"Co-training dynamics ({nt} trials) — {dataset_name}",
        fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout()
    return fig


def plot_final_bars(cv_results, colors=None, dataset_name=""):
    """Final test-set AUC & AP bar chart (solo vs collab vs GRU)."""
    apply_pub_style()
    colors = colors or COLORS
    nt = cv_results["n_trials"]

    # (label, auc_arr, ap_arr, color, hatch)
    methods = []
    for name in cv_results["model_names"]:
        methods.append((f"Solo\n{name}",
                        np.array(cv_results["solo"][name]["auc"]),
                        np.array(cv_results["solo"][name]["ap"]),
                        colors["solo"], ""))
    for name in cv_results["model_names"]:
        methods.append((f"Collab\n{name}",
                        np.array(cv_results["collab"][name]["auc"]),
                        np.array(cv_results["collab"][name]["ap"]),
                        colors["collab"], "//"))
    methods.append(("Collab\nensemble",
                    np.array(cv_results["collab"]["ensemble"]["auc"]),
                    np.array(cv_results["collab"]["ensemble"]["ap"]),
                    colors["ensemble"], "//"))

    ga = np.array(cv_results["gru"]["auc"])
    gp = np.array(cv_results["gru"]["ap"])
    if not np.all(np.isnan(ga)):
        methods.append(("GRU\njudge", ga, gp, colors["gru"], "xx"))

    n_m = len(methods)
    rng = np.random.default_rng(58)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.0 * n_m), 4.5))

    for ax_idx, metric in enumerate(["AUC", "AP"]):
        ax = axes[ax_idx]
        for i, (lbl, auc_a, ap_a, col, hatch) in enumerate(methods):
            vals = auc_a if metric == "AUC" else ap_a
            m = np.nanmean(vals)
            q25, q75 = np.nanpercentile(vals, [25, 75])

            ax.bar(i, m, 0.62, color=col, edgecolor="white",
                   linewidth=1.2, alpha=0.82, hatch=hatch, zorder=2)
            ax.errorbar(i, m, yerr=[[m - q25], [q75 - m]], fmt="none",
                        ecolor="#333", capsize=4, capthick=1.2,
                        linewidth=1.2, zorder=4)

            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                       color=col, edgecolor="#333", linewidth=0.4,
                       s=18, alpha=0.55, zorder=5)
            ax.text(i, m + 0.025, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color="#333")

        ax.set_xticks(range(n_m))
        ax.set_xticklabels([m[0] for m in methods], fontsize=7.5)
        ax.set_ylabel(f"test {metric}"); ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Test-set {metric}", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        _panel_label(ax, ax_idx)

    axes[1].text(1.0, -0.18,
                 "bar = mean | error = IQR | dots = trials",
                 transform=axes[1].transAxes, fontsize=6,
                 ha="right", va="top", color="#666", style="italic")
    fig.suptitle(
        f"Final test-set performance ({nt} trials) — {dataset_name}",
        fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout()
    return fig


def plot_loss_evolution(cv_results, colors=None, dataset_name="",
                        has_recurrent=True):
    """Simpler loss evolution: IQR ribbon, solo vs collab side-by-side."""
    apply_pub_style()
    colors = colors or COLORS
    model_names = cv_results["model_names"]
    n_rows = len(model_names) + (1 if has_recurrent else 0)

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.2 * n_rows), squeeze=False)

    for r, name in enumerate(model_names):
        for c, lk in enumerate(["train_loss", "val_loss"]):
            ax = axes[r, c]
            _ribbon(ax, _pad_and_stack(cv_results["solo"][name][lk]),
                    colors.get(name, "#555"), f"solo {name}",
                    ls="--", sparse_fade=False)
            _ribbon(ax, _pad_and_stack(cv_results["collab"][name][lk]),
                    colors["collab"], f"collab {name}",
                    ls="-", sparse_fade=False)
            ax.set_title(f"{name} — {'train' if c == 0 else 'val'} loss",
                         fontweight="bold")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            ax.legend(fontsize=6, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc")
            _panel_label(ax, r * 2 + c)

    if has_recurrent:
        r_gru = len(model_names)
        for c, lk in enumerate(["train_loss", "val_loss"]):
            ax = axes[r_gru, c]
            _ribbon(ax, _pad_and_stack(cv_results["gru"][lk]),
                    colors["gru"], "GRU judge", ls="-", sparse_fade=False)
            ax.set_title(f"GRU — {'train' if c == 0 else 'val'} loss",
                         fontweight="bold")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss")
            ax.legend(fontsize=6, frameon=True, framealpha=0.9,
                      edgecolor="#cccccc")
            _panel_label(ax, r_gru * 2 + c)

    fig.suptitle(
        f"Loss evolution ({cv_results['n_trials']} trials) — {dataset_name}",
        fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def save_figures(figures, filepath, project_root=None):
    """Save {tag: fig} dict as PNGs using the filepath stem."""
    from pathlib import Path
    filepath = Path(filepath)
    for tag, fig in figures.items():
        out = filepath.with_suffix(f".{tag}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        rel = out.relative_to(project_root) if project_root else out
        print(f"✓ {tag:16s} → {rel}")
# plotting utility metadata update
