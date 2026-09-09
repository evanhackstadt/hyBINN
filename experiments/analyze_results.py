"""
analyze_results.py

Generated using Claude Sonnett 4.6

=============================================
Generates four key outputs from completed experiment runs:

  1.  ablation_table.csv       — mean ± std test C-index per model config
  2.  cindex_boxplot.png       — box-and-whisker over seeds per config
  3.  kaplan_meier.png         — KM curves (low vs high risk) with log-rank p-value
  4.  pathway_rankings.csv     — aggregated pathway activations across seeds,
                                 with decoded pathway names

Usage:
    python analyze_results.py \
        --runs_dir     experiments/runs \
        --reactome     data/reactome/Ensembl2Reactome_All_Levels.txt \
        --predictions  experiments/runs/full_hybinn/seed_0/predictions_test.csv \
        --out_dir      experiments/figures

    # To pool predictions across all seeds of the best model:
        --predictions "experiments/runs/full_hybinn/seed_*/predictions_test.csv"

Notes on design decisions:
  - Box plot uses seed-level test C-indices (n=10 per config)
  - KM split uses the MEDIAN risk score
  - Pathway aggregation uses mean activation across seeds. If a pathway appears in
    only some seeds' top-10, it still contributes — missing seeds contribute 0.
"""

import argparse
import glob
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


# ── Config ──────────────────────────────────────────────────────────────────

# Display labels and plot order for each model configuration
CONFIG_ORDER = ["binn", "gene", "clinical",
                "binn_gene", "binn_clinical", "gene_clinical", "full_hybinn",
                "baseline_coxph_gene", "baseline_coxph_clinical", "baseline_coxph_combined"]

CONFIG_LABELS = {
    "binn":                    "BINN",
    "gene":                    "Gene MLP",
    "clinical":                "Clinical",
    "binn_gene":               "BINN + Gene",
    "binn_clinical":           "BINN + Clinical",
    "gene_clinical":           "Gene + Clinical",
    "full_hybinn":             "Full HyBINN",
    "baseline_coxph_gene":     "Baseline CoxPH\n(gene)",
    "baseline_coxph_clinical": "Baseline CoxPH\n(clinical)",
    "baseline_coxph_combined": "Baseline CoxPH\n(combined)",
}

# Seaborn palette — one colour per config, consistent across all figures
PALETTE = sns.color_palette("Set2", n_colors=len(CONFIG_ORDER))
CONFIG_COLORS = {cfg: PALETTE[i] for i, cfg in enumerate(CONFIG_ORDER)}

ALL_TASKS = [
    "ablation_table",
    "cindex_boxplot",
    "km_binary",
    "km_3group",
    "pathway_rankings",
    "pathway_barplot",
    "statistical_tests"
]
TASK_ALIASES = {
    "all": ALL_TASKS,
    "km": ["km_binary", "km_3group"],
}
KNOWN_TASKS = sorted(set(ALL_TASKS) | set(TASK_ALIASES.keys()))


def expand_tasks(tasks):
    expanded = []
    for task in tasks:
        if task in TASK_ALIASES:
            for subtask in TASK_ALIASES[task]:
                if subtask not in expanded:
                    expanded.append(subtask)
        elif task not in expanded:
            expanded.append(task)
    return expanded


def is_enabled(task, enabled_tasks):
    return task in enabled_tasks


# legacy top-level baseline loader removed; baseline CoxPH results are stored
# per-seed in `runs/baseline_coxph_<variant>/seed_<n>/baseline_coxph_results.json`.


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_all_results(runs_dir, configs=None):
    """
    Walks runs_dir/{config}/seed_*/results.json and returns a flat DataFrame.
    Each row is one (config, seed) pair.
    """
    configs = configs or CONFIG_ORDER
    records = []

    for cfg_name in configs:
        model_dir = os.path.join(runs_dir, cfg_name)
        if not os.path.isdir(model_dir):
            print(f"  [MISSING] No directory for config: {cfg_name}")
            continue

        for seed_dir in sorted(os.listdir(model_dir)):
            results_path = os.path.join(model_dir, seed_dir, "results.json")

            if os.path.exists(results_path):
                with open(results_path) as f:
                    r = json.load(f)

                row = {
                    "config":           cfg_name,
                    "seed_dir":         seed_dir,
                    "seed":             r.get("seed"),
                    "test_cindex":      r.get("test_cindex"),
                    "cv_mean_cindex":   r.get("cv_mean_cindex"),
                    "boot_lower":       (r.get("bootstrap_ci") or {}).get("lower"),
                    "boot_upper":       (r.get("bootstrap_ci") or {}).get("upper"),
                    "boot_mean":        (r.get("bootstrap_ci") or {}).get("mean"),
                    "n_test":           r.get("n_test"),
                    "top_pathways":     r.get("top_pathways", {}),
                }

                # Branch weights, if saved
                for branch, w in (r.get("branch_weights") or {}).items():
                    row[f"weight_{branch}"] = w

                records.append(row)
                continue

            # If no standard results.json, check for baseline CoxPH per-seed JSON
            alt_path = os.path.join(model_dir, seed_dir, "baseline_coxph_results.json")
            if os.path.exists(alt_path):
                with open(alt_path) as f:
                    data = json.load(f)

                models = data.get("models", [])
                if models:
                    m = models[0]
                    try:
                        seed_val = int(seed_dir.replace("seed_", ""))
                    except Exception:
                        seed_val = m.get("model")

                    row = {
                        "config":           cfg_name,
                        "seed_dir":         seed_dir,
                        "seed":             seed_val,
                        "test_cindex":      m.get("c_index", m.get("test_cindex")),
                        "cv_mean_cindex":   None,
                        "boot_lower":       None,
                        "boot_upper":       None,
                        "boot_mean":        None,
                        "n_test":           m.get("n_test"),
                        "top_pathways":     {},
                    }
                    records.append(row)
            else:
                print(f"  [MISSING] {cfg_name}/{seed_dir}/results.json")

    df = pd.DataFrame(records)

    print(f"\nLoaded {len(df)} runs across {df['config'].nunique()} configs.")
    return df


def load_predictions(paths):
    """
    Loads and concatenates predictions_test.csv files.
    Expects columns: patient_id, risk_final, time, event.
    If multiple files are passed, pools all patients (useful for multi-seed aggregation,
    but be aware this inflates n — use single-seed for a cleaner KM figure).
    """
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"  [MISSING] {p}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def resolve_prediction_paths(predictions, runs_dir, km_model, km_seed):
    """Resolve KM prediction file paths from explicit files or default km_model/km_seed."""
    if predictions:
        resolved = []
        for pattern in predictions:
            if any(ch in pattern for ch in "*?[]"):
                resolved.extend(sorted(glob.glob(pattern)))
            elif os.path.exists(pattern):
                resolved.append(pattern)
            else:
                print(f"  [MISSING] {pattern}")
        return list(dict.fromkeys(resolved))

    if km_seed == "all":
        return sorted(glob.glob(os.path.join(runs_dir, km_model, "seed_*", "predictions_test.csv")))

    return [os.path.join(runs_dir, km_model, km_seed, "predictions_test.csv")]


# ── 1. Ablation Table ────────────────────────────────────────────────────────

def make_ablation_table(results_df, out_dir):
    """
    Produces ablation_table.csv with mean ± std test C-index per config,
    sorted by CONFIG_ORDER, plus CV C-index and bootstrap 95% CI columns.
    """
    rows = []
    for cfg in CONFIG_ORDER:
        sub = results_df[results_df["config"] == cfg]
        if sub.empty:
            continue

        test_ci = sub["test_cindex"].dropna()
        cv_ci   = sub["cv_mean_cindex"].dropna()

        row = {
            "Model":                CONFIG_LABELS.get(cfg, cfg),
            "N seeds":              len(test_ci),
            "CV C-index (mean±std)": f"{cv_ci.mean():.4f} ± {cv_ci.std():.4f}" if len(cv_ci) else "—",
            "Test C-index (mean±std)": f"{test_ci.mean():.4f} ± {test_ci.std():.4f}" if len(test_ci) else "—",
            # Bootstrap CI: average the per-seed lower/upper (interpretable as
            # typical uncertainty around a single test estimate for this model)
            "Bootstrap 95% CI":    (
                f"[{sub['boot_lower'].mean():.4f}, {sub['boot_upper'].mean():.4f}]"
                if sub["boot_lower"].notna().any() else "—"
            ),
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "ablation_table.csv")
    table.to_csv(out_path, index=False)
    print(f"\nAblation table saved → {out_path}")
    print(table.to_string(index=False))
    return table


# ── 2. Box-and-Whisker Plot ──────────────────────────────────────────────────

def make_boxplot(results_df, out_dir):
    """
    Box-and-whisker of test C-index across seeds, one box per model config.
    Each point is one (config, seed) test C-index — n=10 per box.
    """
    # Filter to configs present in data, preserve CONFIG_ORDER
    present = [c for c in CONFIG_ORDER if c in results_df["config"].values]
    plot_df = results_df[results_df["config"].isin(present)].copy()
    plot_df["config"] = pd.Categorical(plot_df["config"], categories=present, ordered=True)
    plot_df = plot_df.sort_values("config")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw boxes
    bp = ax.boxplot(
        [plot_df[plot_df["config"] == c]["test_cindex"].dropna().values for c in present],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, linestyle="none"),
        widths=0.5,
    )
    for patch, cfg in zip(bp["boxes"], present):
        patch.set_facecolor(CONFIG_COLORS[cfg])
        patch.set_alpha(0.75)

    # Overlay individual seed points (jittered)
    rng = np.random.default_rng(42)
    for i, cfg in enumerate(present, start=1):
        vals = plot_df[plot_df["config"] == cfg]["test_cindex"].dropna().values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals,
                   color=CONFIG_COLORS[cfg], edgecolors="black",
                   linewidths=0.5, s=30, zorder=3, alpha=0.9)

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in present],
                       fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Test C-index", fontsize=11)
    # ax.set_title("Test C-index across 10 random seeds — HyBINN ablation", fontsize=12)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "cindex_boxplot.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Box plot saved → {out_path}")


# ── 3. Kaplan-Meier Curves ───────────────────────────────────────────────────

def make_km_plot_binary(predictions_df, out_dir, label="Full HyBINN"):
    """
    Splits patients into 2 risk groups (low/high) and plots KM survival curves.

    Args:
        predictions_df: DataFrame with columns [risk_final, time, event]
        label: plot title label (model config name)
    """
    df = predictions_df[["risk_final", "time", "event"]].dropna()

    threshold = df["risk_final"].median()

    high_risk = df[df["risk_final"] >= threshold]
    low_risk  = df[df["risk_final"] <  threshold]

    n_high = len(high_risk)
    n_low  = len(low_risk)

    # Log-rank test
    lr = logrank_test(
        durations_A=high_risk["time"],  events_A=high_risk["event"],
        durations_B=low_risk["time"],   events_B=low_risk["event"],
    )
    p_val = lr.p_value

    # Hazard Ratio (fit univariate Cox model with binary risk group covariate
    df = predictions_df[["risk_final", "time", "event"]].dropna().copy()

    # Binary covariate: 1 = high risk, 0 = low risk
    # CoxPHFitter will give HR for a one-unit increase in this covariate,
    # i.e., the HR of high-risk vs. low-risk group directly.
    df["high_risk"] = (df["risk_final"] >= threshold).astype(int)

    cph = CoxPHFitter()
    cph.fit(df[["high_risk", "time", "event"]],
            duration_col="time", event_col="event")

    summary = cph.summary
    hr       = float(summary.loc["high_risk", "exp(coef)"])
    hr_lower = float(summary.loc["high_risk", "exp(coef) lower 95%"])
    hr_upper = float(summary.loc["high_risk", "exp(coef) upper 95%"])
    p_val    = float(summary.loc["high_risk", "p"])

    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
    print(f"  Hazard Ratio: {hr:.3f} (95% CI: {hr_lower:.3f}-{hr_upper:.3f}), {p_str}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(high_risk["time"], high_risk["event"], label=f"High risk  (n={n_high})")
    kmf_low.fit(low_risk["time"],   low_risk["event"],   label=f"Low risk   (n={n_low})")

    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="firebrick",
                                    linewidth=2)
    kmf_low.plot_survival_function( ax=ax, ci_show=True, color="steelblue",
                                    linewidth=2)

    # Annotate plot
    p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
    ax.text(0.97, 0.97,
            f"Log-rank {p_str}\nHR = {hr:.3f} (95% CI: {hr_lower:.3f}-{hr_upper:.3f})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(f"Kaplan-Meier (split on median risk score)\n{label}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(which="major", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "kaplan_meier.png")
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Low/High KM plot saved → {out_path}  (log-rank {p_str})")


def make_km_plot_3group(predictions_df, out_dir, label="Full HyBINN"):
    """
    Three-group KM plot split on tertiles (33rd / 66th percentile of risk score).
    Use tertiles rather than arbitrary thresholds — maximizes balance and is
    the standard approach in survival stratification literature.

    NOTE on pooled predictions: if predictions_df pools multiple seeds, the
    curves are smoother but confidence bands are artificially narrow (effective
    n = single test set). Appropriate for visualization; do not report pooled
    log-rank p as if n = n_pooled.
    """
    from lifelines import CoxPHFitter

    df = predictions_df[["risk_final", "time", "event"]].dropna().copy()

    t33 = df["risk_final"].quantile(0.333)
    t67 = df["risk_final"].quantile(0.667)

    df["risk_group"] = pd.cut(
        df["risk_final"],
        bins=[-np.inf, t33, t67, np.inf],
        labels=["Low risk", "Moderate risk", "High risk"]
    )

    n_low  = (df["risk_group"] == "Low risk").sum()
    n_mod  = (df["risk_group"] == "Moderate risk").sum()
    n_high = (df["risk_group"] == "High risk").sum()

    # Pairwise log-rank tests (low vs high is the primary comparison)
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    # Omnibus test across all three groups
    omnibus = multivariate_logrank_test(
        df["time"], df["risk_group"], df["event"]
    )
    p_omni = omnibus.p_value

    # Primary pairwise: low vs high
    lh = logrank_test(
        df[df["risk_group"] == "Low risk"]["time"],
        df[df["risk_group"] == "High risk"]["time"],
        df[df["risk_group"] == "Low risk"]["event"],
        df[df["risk_group"] == "High risk"]["event"],
    )

    # Cox HR: use ordinal encoding (0/1/2) to get a single HR per one-tier step
    df["risk_ordinal"] = df["risk_group"].map(
        {"Low risk": 0, "Moderate risk": 1, "High risk": 2}
    )
    cph = CoxPHFitter()
    cph.fit(df[["risk_ordinal", "time", "event"]],
            duration_col="time", event_col="event")
    summary  = cph.summary
    hr       = float(summary.loc["risk_ordinal", "exp(coef)"])
    hr_lower = float(summary.loc["risk_ordinal", "exp(coef) lower 95%"])
    hr_upper = float(summary.loc["risk_ordinal", "exp(coef) upper 95%"])

    print(f"  Omnibus log-rank p = {p_omni:.4f}")
    print(f"  Low vs High log-rank p = {lh.p_value:.4f}")
    print(f"  Per-tier HR = {hr:.3f} (95% CI: {hr_lower:.3f}–{hr_upper:.3f})")

    # ── Plot ──────────────────────────────────────────────────────
    colors = {"Low risk": "steelblue", "Moderate risk": "goldenrod", "High risk": "firebrick"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for group, color in colors.items():
        sub = df[df["risk_group"] == group]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event"],
                label=f"{group}  (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

    p_omni_str = "p < 0.001" if p_omni < 0.001 else f"p = {p_omni:.4f}"
    p_lh_str   = "p < 0.001" if lh.p_value < 0.001 else f"p = {lh.p_value:.4f}"

    annotation = (
        f"Omnibus log-rank {p_omni_str}\n"
        f"Low vs High {p_lh_str}\n"
        f"Per-tier HR = {hr:.3f} ({hr_lower:.3f}–{hr_upper:.3f})"
    )
    ax.text(0.97, 0.97, annotation,
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(f"Kaplan-Meier (tertile split)\n{label}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(which="major", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "kaplan_meier_3group.png")
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  3-group KM saved → {out_path}")


# ── 4. OLD Pathway Analysis ──────────────────────────────────────────────────────
# based on mean activation; replaced by integrated gradients in pathway_attribution.py

'''
def build_reactome_name_map(reactome_path):
    """
    Reads the Ensembl2Reactome TSV and returns {ReactomePathwayID: PathwayName}.
    """
    df = pd.read_csv(reactome_path, delimiter="\t",
                     names=["EnsemblID", "ReactomePathwayID", "URL",
                            "PathwayName", "Evidence", "Species"])
    df = df[df["Species"] == "Homo sapiens"]
    return df.drop_duplicates("ReactomePathwayID").set_index("ReactomePathwayID")["PathwayName"].to_dict()


def aggregate_pathways(results_df, reactome_path=None, out_dir="."):
    """
    Aggregates top_pathways dicts across all (config, seed) pairs.
    Reports mean activation and frequency of appearance in top-10 per run.

    If reactome_path is provided, decodes pathway IDs to human-readable names.
    """
    # Collect all pathway activation values across all runs
    all_activations = {}   # {pathway_id: [activation_values]}
    n_runs = 0

    for _, row in results_df.iterrows():
        tp = row.get("top_pathways")
        if not tp or not isinstance(tp, dict):
            continue
        n_runs += 1
        for pid, val in tp.items():
            all_activations.setdefault(pid, []).append(float(val))

    if not all_activations:
        print("  No pathway data found in results.")
        return None

    # Decode names
    name_map = {}
    if reactome_path and os.path.exists(reactome_path):
        name_map = build_reactome_name_map(reactome_path)
    else:
        print("  [NOTE] No Reactome file provided or found — pathway IDs will not be decoded.")

    records = []
    for pid, vals in all_activations.items():
        records.append({
            "pathway_id":       pid,
            "pathway_name":     name_map.get(pid, pid),   # fall back to ID if not found
            "mean_activation":  np.mean(vals),
            "std_activation":   np.std(vals),
            "n_appearances":    len(vals),        # how many runs had this in their top-10
            "pct_appearances":  100.0 * len(vals) / n_runs,
        })

    pathway_df = pd.DataFrame(records).sort_values("mean_activation", ascending=False)

    out_path = os.path.join(out_dir, "pathway_rankings.csv")
    pathway_df.to_csv(out_path, index=False)
    print(f"\nPathway rankings saved → {out_path}")

    # Print top 20
    print(f"\nTop 20 pathways by mean activation (across {n_runs} runs):")
    print(pathway_df.head(20).to_string(index=False))

    return pathway_df


def make_pathway_barplot(pathway_df, out_dir, top_n=20):
    """
    Horizontal bar chart of top-N pathways by mean activation.
    Bar width = mean activation, error bars = std across seeds.
    Color encodes % of runs the pathway appeared in (frequency).
    """
    df = pathway_df.head(top_n).copy()
    # Truncate long names for readability
    df["label"] = df["pathway_name"].apply(lambda x: x[:55] + "…" if len(str(x)) > 55 else x)
    df = df[::-1]  # reverse so top pathway is at top of plot

    norm = plt.Normalize(0, 100)
    cmap = plt.cm.YlOrRd
    colors = [cmap(norm(v)) for v in df["pct_appearances"]]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
    bars = ax.barh(df["label"], df["mean_activation"],
                   xerr=df["std_activation"], color=colors,
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(elinewidth=0.8, capsize=2))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("% runs in top-10", fontsize=9)

    ax.set_xlabel("Mean pathway activation (± std)", fontsize=10)
    ax.set_title(f"Top {top_n} Reactome pathways by mean activation\nacross all HyBINN runs", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "pathway_barplot.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Pathway bar plot saved → {out_path}")
'''


# ── 5. Statistical Tests ─────────────────────────────────────────────────────

def make_statistical_tests(results_df, out_dir):
    """
    Friedman test across all model configs, followed by pairwise Wilcoxon
    signed-rank post-hoc tests with Holm correction if omnibus p < 0.05.

    Uses seed as the blocking variable (paired across configs), since the same
    seed index produces the same train/test split for all configurations.

    Saves:
        statistical_tests.txt     — omnibus result + pairwise table
        pairwise_pvalues.csv      — raw and corrected p-values
        pairwise_heatmap.png      — heatmap of corrected p-values
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations

    # ── Build [seeds x configs] matrix ───────────────────────────────────────
    present_configs = [c for c in CONFIG_ORDER if c in results_df["config"].values]

    pivot = (
        results_df[results_df["config"].isin(present_configs)]
        .pivot(index="seed", columns="config", values="test_cindex")
        [present_configs]   # enforce CONFIG_ORDER column order
        .dropna()
    )

    n_seeds, n_configs = pivot.shape
    print(f"\n  Paired matrix: {n_seeds} seeds × {n_configs} configs")

    # ── Friedman test ─────────────────────────────────────────────────────────
    friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot[c].values for c in present_configs])
    print(f"  Friedman χ²({n_configs - 1}) = {friedman_stat:.4f}, p = {friedman_p:.4f}")

    lines = [
        "=" * 60,
        "Omnibus: Friedman Test",
        f"  χ²({n_configs - 1}) = {friedman_stat:.4f},  p = {friedman_p:.4f}",
        f"  n = {n_seeds} paired seeds",
        "",
    ]

    # ── Pairwise Wilcoxon signed-rank (always run, flag if omnibus NS) ────────
    if friedman_p >= 0.05:
        lines.append("  NOTE: Omnibus test not significant (p ≥ 0.05).")
        lines.append("  Pairwise tests reported for completeness; interpret cautiously.")
    lines += ["", "Pairwise: Wilcoxon Signed-Rank (Holm-corrected)", "-" * 60]

    pairs = list(combinations(present_configs, 2))
    raw_p = []
    for a, b in pairs:
        _, p = stats.wilcoxon(pivot[a].values, pivot[b].values)
        raw_p.append(p)

    reject, p_corrected, _, _ = multipletests(raw_p, method="holm")

    records = []
    for (a, b), p_raw, p_corr, rej in zip(pairs, raw_p, p_corrected, reject):
        records.append({
            "config_A":    CONFIG_LABELS.get(a, a),
            "config_B":    CONFIG_LABELS.get(b, b),
            "p_raw":       round(p_raw, 6),
            "p_holm":      round(p_corr, 6),
            "significant": rej,
        })

    pairs_df = pd.DataFrame(records)
    pairs_df_str = pairs_df.to_string(index=False)
    lines.append(pairs_df_str)
    lines += ["", "=" * 60]

    report = "\n".join(lines)
    print("\n" + report)

    # ── Save text report ──────────────────────────────────────────────────────
    txt_path = os.path.join(out_dir, "statistical_tests.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"\n  Statistical test report → {txt_path}")

    # ── Save pairwise CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "pairwise_pvalues.csv")
    pairs_df.to_csv(csv_path, index=False)
    print(f"  Pairwise p-values → {csv_path}")

    # ── Heatmap of Holm-corrected p-values ────────────────────────────────────
    labels = [CONFIG_LABELS.get(c, c) for c in present_configs]
    n = len(labels)
    matrix = np.ones((n, n))

    for (a, b), p_corr in zip(pairs, p_corrected):
        i = present_configs.index(a)
        j = present_configs.index(b)
        matrix[i, j] = p_corr
        matrix[j, i] = p_corr

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn_r")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Holm-corrected p-value", fontsize=9)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            txt = f"{matrix[i,j]:.3f}"
            color = "white" if matrix[i, j] < 0.3 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Pairwise Wilcoxon p-values (Holm-corrected)", fontsize=11)
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "pairwise_heatmap.png")
    fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {heatmap_path}")

    return pairs_df



# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HyBINN final analysis")
    p.add_argument("--runs_dir",    required=True,
                   help="Path to experiments/runs/")
    p.add_argument("--reactome",    default=None,
                   help="Path to Ensembl2Reactome_All_Levels.txt (for pathway decoding)")
    p.add_argument("--predictions", nargs="+", default=None,
                   help=("Path(s) to predictions_test.csv for KM plot. "
                         "Supports explicit files or glob patterns. "
                         "If omitted, the script will use --km_model and --km_seed "
                         "to infer the default run path(s)."))
    p.add_argument("--km_model",    default="full_hybinn",
                   help="Which model config to use for KM plot when --predictions is omitted.")
    p.add_argument("--km_seed",     default="seed_0",
                   help="Which seed dir to use for KM plot when --predictions is omitted. "
                        "Set to 'all' to pool all seeds (inflates n — use with care).")
    p.add_argument("--configs",     nargs="+", default=None,
                   help="Subset of configs to analyze (default: all)")
    p.add_argument("--tasks",       nargs="+", default=["all"],
                   choices=KNOWN_TASKS,
                   help=("Which outputs to produce. Choices: all, km, "
                         "ablation_table, cindex_boxplot, km_binary, "
                         "km_3group, pathway_rankings, pathway_barplot. "
                         "Use 'all' for every output, or 'km' for both KM plots."))
    p.add_argument("--out_dir",     default="figures",
                   help="Output directory for figures and tables (default: figures/)")
    return p.parse_args()


def main():
    args = parse_args()
    args.tasks = expand_tasks(args.tasks)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("HyBINN Final Analysis")
    print("=" * 60)
    print(f"Enabled tasks: {', '.join(args.tasks)}")

    # ── Load all results ──────────────────────────────────────────
    results_df = load_all_results(args.runs_dir, configs=args.configs)

    if results_df.empty:
        print("No results loaded. Check --runs_dir.")
        return

    if is_enabled("ablation_table", args.tasks):
        print("\n── 1. Ablation Table ──")
        make_ablation_table(results_df, args.out_dir)

    if is_enabled("cindex_boxplot", args.tasks):
        print("\n── 2. Box-and-Whisker Plot ──")
        make_boxplot(results_df, args.out_dir)

    if any(is_enabled(t, args.tasks) for t in ("km_binary", "km_3group")):
        print("\n── 3. Kaplan-Meier Curves ──")

        pred_paths = resolve_prediction_paths(
            args.predictions, args.runs_dir, args.km_model, args.km_seed
        )

        if pred_paths:
            print(f"  Using {len(pred_paths)} prediction file(s) for KM plot.")
            preds_df = load_predictions(pred_paths)
            if preds_df is not None:
                label = CONFIG_LABELS.get(args.km_model, args.km_model)
                if is_enabled("km_binary", args.tasks):
                    make_km_plot_binary(preds_df, args.out_dir, label=label)
                if is_enabled("km_3group", args.tasks):
                    make_km_plot_3group(preds_df, args.out_dir, label=label)
        else:
            print("  No prediction files found. Skipping KM plot.")
            print("  Tip: ensure predictions_test.csv exists in your run directories,")
            print("       or pass --predictions path/to/predictions_test.csv")

    '''
    if any(is_enabled(t, args.tasks) for t in ("pathway_rankings", "pathway_barplot")):
        print("\n── 4. Pathway Analysis ──")

        # Filter to BINN-containing configs (only those save top_pathways)
        binn_configs = [c for c in CONFIG_ORDER if "binn" in c]
        binn_results = results_df[results_df["config"].isin(binn_configs)]

        if binn_results.empty:
            print("  No BINN results found. Skipping pathway analysis.")
        else:
            pathway_df = aggregate_pathways(binn_results, args.reactome, args.out_dir)
            if pathway_df is not None and is_enabled("pathway_barplot"):
                make_pathway_barplot(pathway_df, args.out_dir, top_n=20)
    '''
    
    if is_enabled("statistical_tests", args.tasks):
        print("\n── 5. Statistical Tests ──")
        make_statistical_tests(results_df, args.out_dir)
    
    print("\n── Done ──")
    print(f"All outputs saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
