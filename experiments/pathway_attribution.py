# experiments/pathway_attribution.py
"""
pathway_attribution.py

Computes pathway-level feature importance for a trained BINN branch using
Integrated Gradients (Sundararajan, Taly & Yan, 2017), evaluated POST-HOC
on a frozen, already-trained HyBINN checkpoint. No retraining required.

--------------------------------------------------------------------------
Why not mean activation, why not raw partial derivatives?
--------------------------------------------------------------------------
Mean pathway activation (post-tanh) measures how "on" a pathway node is on
average -- not how much that pathway's signal actually moves the model's
output. Raw partial derivatives (the approach used by CoxPAS-Net, Hao et
al. 2018) fix that "on vs. relevant" conflation, but suffer from gradient
saturation in tanh-heavy architectures like this one: a saturated unit can
be highly influential yet show ~0 *local* gradient at the observed point.

Integrated Gradients accumulates the gradient along the entire straight-line
path from a reference baseline to the observed activation, rather than
evaluating it once at the endpoint. This is immune to endpoint saturation,
and satisfies a completeness axiom (attributions sum exactly to
f(x) - f(baseline)), giving a built-in correctness check via
`convergence_delta`.

This mirrors P-NET (Elmarakeby et al., Nature 2021) -- the closest prior
BINN to this one, also Reactome-pathway-structured -- which used DeepLIFT
for the identical reason (avoiding saturation/discontinuity artifacts in
raw-gradient attribution). IG and DeepLIFT are both baseline-relative
attribution methods; IG is used here because captum ships it as a drop-in
wrapper requiring no custom backward rules.

--------------------------------------------------------------------------
Why this is NOT captum.attr.LayerIntegratedGradients
--------------------------------------------------------------------------
BINNBranch reuses a single nn.Tanh() instance (self.tanh) at three
different points in its forward pass (pathway layer, hidden layer, output
layer). captum's LayerIntegratedGradients hooks a *module instance* and
cannot disambiguate which of the three invocations per forward call you
want attributions for. To avoid this, the frozen model is manually split
into two callables:

    (a) x_mapped              -> pathway_activations   (sc1 + mask + tanh)
    (b) pathway_activations   -> model output           (rest of BINN,
                                                          fusion, survival head)

and vanilla captum.attr.IntegratedGradients is applied to (b), with the
other branches' outputs held fixed as additional forward arguments (they
don't depend on pathway_activations, so holding them fixed is exact, not
an approximation). This is mathematically identical to attributing to an
intermediate layer, without depending on captum's module-hook machinery.

--------------------------------------------------------------------------
Baseline choice
--------------------------------------------------------------------------
The baseline is the all-zero pathway-activation vector -- i.e. "no pathway
signal reaches the rest of the network." This is the standard reference
point for tanh-space attribution (mirrors DeepLIFT's typical all-zero
reference in P-NET). Report this choice explicitly in the write-up: IG
attributions are always relative to a stated counterfactual, not absolute.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    pip install captum --break-system-packages   # if not already installed

    python pathway_attribution.py \
        --run_dir experiments/runs/full_hybinn/seed_0 \
        --out_dir experiments/figures

Requires that <run_dir> contains best_model.pt and config_frozen.yaml,
i.e. it must be a run directory produced by train_hybinn.py.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from utils.seed import set_seed
from utils.config import load_config
from utils.splitting import stratified_train_test_split
from utils.logging import get_logger

from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from datasets.dataset import SurvivalDataset, get_dataloader

from train_hybinn import instantiate_model


# ---- Reactome ID -> human-readable name decoding (mirrors analyze_results.py) ----

def build_reactome_name_map(reactome_path):
    """
    Reads the Ensembl2Reactome TSV and returns {ReactomePathwayID: PathwayName}.
    """
    df = pd.read_csv(reactome_path, delimiter="\t",
                     names=["EnsemblID", "ReactomePathwayID", "URL",
                            "PathwayName", "Evidence", "Species"])
    df = df[df["Species"] == "Homo sapiens"]
    return df.drop_duplicates("ReactomePathwayID").set_index("ReactomePathwayID")["PathwayName"].to_dict()


# ---- Model surgery: split the BINN branch around its pathway layer ----

def _make_pathway_fns(binn_branch):
    """
    Args:
        binn_branch (BINNBranch): the frozen, trained BINN branch (model.branches['binn'])

    Returns:
        get_pathway_activations(x_mapped) -> pathway_activations
            Reproduces BINNBranch's sc1 + mask + tanh step exactly.
        binn_tail(pathway_activations) -> binn branch output
            Reproduces everything in BINNBranch AFTER the pathway layer.
            Safe to call in eval mode (dropout becomes a no-op).
    """

    def get_pathway_activations(x_mapped):
        masked_weights = binn_branch.sc1.weight * binn_branch.mask.T
        return binn_branch.tanh(F.linear(x_mapped, masked_weights, binn_branch.sc1.bias))

    def binn_tail(pathway_activations):
        x = binn_branch.dropout(pathway_activations)
        x = binn_branch.tanh(binn_branch.fc2(x))
        x = binn_branch.dropout(x)
        x = binn_branch.tanh(binn_branch.fc3(x))
        return x

    return get_pathway_activations, binn_tail


# ---- Main attribution routine ----

def compute_pathway_attributions(model, dataloader, device, n_steps=50, logger=None):
    """
    Runs Integrated Gradients on the BINN branch's pathway layer across an
    entire dataloader (typically the frozen run's own held-out test set).

    Args:
        model (HyBINN): frozen, trained model, already .eval()'d with weights loaded
        dataloader (DataLoader): should be built from the SAME split used to
                                  train/evaluate this checkpoint (see main())
        device (torch.device)
        n_steps (int): number of interpolation steps along the IG path.
                        Increase if convergence_delta is large relative to
                        the risk score's scale.

    Returns:
        mean_abs_attr (ndarray): shape (n_pathways,) -- mean |IG attribution|
                                  per pathway across all patients in dataloader
        all_attrs (ndarray): shape (n_patients, n_pathways) -- per-patient
                              signed attributions, kept for downstream analysis
                              (e.g. do high-risk vs low-risk patients rely on
                              different pathways?)
        convergence_deltas (list of ndarray): per-batch completeness-axiom
                              sanity check; should be small vs. the model's
                              output scale
    """
    model.eval()
    binn_branch = model.branches['binn']
    full_branch_order = list(model.branches.keys())
    non_binn_names = [n for n in full_branch_order if n != 'binn']

    get_pathway_acts, binn_tail = _make_pathway_fns(binn_branch)

    def pathway_to_output(pathway_activations, *other_tensors):
        """
        Recomputes the model's final output as a function of pathway_activations
        only, with every other branch's output supplied as a fixed extra arg.
        captum auto-expands tensor additional_forward_args to match the
        n_steps-interpolated batch, so this composes correctly with IG.
        """
        binn_out = binn_tail(pathway_activations)
        other_dict = dict(zip(non_binn_names, other_tensors))
        outs = [binn_out if name == 'binn' else other_dict[name] for name in full_branch_order]

        if model.emb_dim == 1:
            weights = torch.softmax(model.fusion_logits, dim=0)
            r_final = 0
            for w, r in zip(weights, outs):
                r_final = r_final + w * r
            return r_final
        else:
            z = model.attention_fusion(outs)
            return model.survival_head(z)

    ig = IntegratedGradients(pathway_to_output)

    all_attrs, convergence_deltas = [], []

    for batch in dataloader:
        x_mapped = batch['X_mapped'].to(device)
        x_unmapped = batch['X_unmapped'].to(device)
        x_clinical = batch['X_clinical'].to(device)

        # Other branches don't depend on pathway_activations -> compute once, hold fixed.
        other_tensors = tuple(
            model.branches[name](x_mapped, x_unmapped, x_clinical).detach()
            for name in non_binn_names
        )

        pathway_acts = get_pathway_acts(x_mapped).detach().requires_grad_(True)
        baseline = torch.zeros_like(pathway_acts)

        attributions, delta = ig.attribute(
            inputs=pathway_acts,
            baselines=baseline,
            additional_forward_args=other_tensors,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        all_attrs.append(attributions.detach().cpu().numpy())
        convergence_deltas.append(delta.detach().cpu().numpy())

        if logger:
            logger.info(f"Batch done. mean |convergence delta| = {np.abs(delta.detach().cpu().numpy()).mean():.6f}")

    all_attrs = np.concatenate(all_attrs, axis=0)     # (n_patients, n_pathways)
    mean_abs_attr = np.abs(all_attrs).mean(axis=0)    # (n_pathways,)

    # Sanity guard: a convergence delta of ~0 is good news (IG converged), but if
    # attributions themselves are ALSO exactly 0 everywhere, that's a red flag for
    # a disconnected graph (e.g. pathway_acts not actually reaching model output)
    # rather than genuine convergence. Distinguish the two explicitly.
    if np.allclose(all_attrs, 0.0):
        if logger:
            logger.warning(
                "WARNING: all IG attributions are exactly 0. A convergence delta "
                "of 0 alongside all-zero attributions suggests a disconnected "
                "computation graph, not genuine convergence -- double check that "
                "pathway_activations actually reaches the model output."
            )

    return mean_abs_attr, all_attrs, convergence_deltas


# ---- Single-run core logic (reused by both single-run and multi-seed modes) ----

def run_single_seed(run_dir, config_path=None, reactome_path_override=None,
                    n_steps=50, out_dir=None, save_outputs=True, logger=None):
    """
    Loads one frozen run (a single (model_config, seed) checkpoint), reproduces
    its exact preprocessing + test split, and computes pathway-level Integrated
    Gradients attributions against it.

    Args:
        run_dir (str): frozen run directory (contains best_model.pt, config_frozen.yaml)
        config_path (str): override path to config_frozen.yaml (default: <run_dir>/config_frozen.yaml)
        reactome_path_override (str): override the reactome path stored in the config,
                                       if the file has moved since training
        n_steps (int): IG interpolation steps
        out_dir (str): where to save this seed's outputs (default: run_dir)
        save_outputs (bool): if True, writes pathway_ig_rankings.csv and the
                              per-patient attribution .npy to out_dir
        logger: shared logger, or None to create a fresh one

    Returns:
        pathway_labels (list[str]): Reactome pathway IDs, in matrix-column order
        mean_abs_attr (ndarray): shape (n_pathways,)
        all_attrs (ndarray): shape (n_patients, n_pathways)
        deltas (list[ndarray]): per-batch IG convergence deltas
        reactome_path (str): the reactome file actually used (for name decoding upstream)
    """
    logger = logger or get_logger(__name__)
    config_path = config_path or os.path.join(run_dir, "config_frozen.yaml")
    out_dir = out_dir or run_dir
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(config_path, os.path.abspath(__file__))
    if reactome_path_override:
        cfg['data']['reactome_path'] = reactome_path_override
    reactome_path = cfg['data']['reactome_path']

    seed = cfg['data']['random_seed']
    set_seed(seed)   # reproduces the EXACT train/val/test split used to train this checkpoint

    logger.info(f"Loading frozen run from {run_dir}")
    logger.info(f"Active branches: {cfg['model']['branches']}")

    if 'binn' not in cfg['model']['branches']:
        raise ValueError(f"{run_dir} has no BINN branch -- nothing to attribute pathways for.")

    # ---- Reproduce preprocessing (deterministic given the same input files) ----
    df = pd.read_csv(cfg['data']['gene_data_path'], index_col=0)
    clinical_df = pd.read_csv(cfg['data']['clinical_data_path'], index_col=0)
    times = df['OS.time'].to_numpy(dtype=np.float32)
    events = df['OS'].to_numpy(dtype=np.float32)
    gene_df = df.drop(columns=['OS.time', 'OS'])

    pathway_map = build_reactome_map(reactome_path)
    mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
    mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)

    x_mapped = gene_df[gene_labels].to_numpy(dtype=np.float32)
    x_unmapped = gene_df[unmapped].to_numpy(dtype=np.float32)
    x_clinical = clinical_df.to_numpy(dtype=np.float32)

    dataset = SurvivalDataset(x_mapped, x_unmapped, x_clinical, times, events)

    # Reproduce the exact test split used at training time (same seed, same StratifiedShuffleSplit)
    _, test_idx = stratified_train_test_split(events, cfg)
    test_dataloader = get_dataloader(dataset, test_idx, cfg, shuffle=False)

    # ---- Rebuild architecture and load frozen weights ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = cfg['model']['embedding_dim']
    model = instantiate_model(cfg, mapped, unmapped, clinical_df.columns.to_list(),
                              mask, pathway_labels, embed_dim)

    best_model_path = os.path.join(run_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---- Run Integrated Gradients ----
    logger.info(f"Running Integrated Gradients (n_steps={n_steps}) on {len(test_idx)} test patients...")
    mean_abs_attr, all_attrs, deltas = compute_pathway_attributions(
        model, test_dataloader, device, n_steps=n_steps, logger=logger
    )

    max_delta = max(np.abs(d).max() for d in deltas)
    logger.info(f"Max |convergence delta| across all batches: {max_delta:.6f}")

    if save_outputs:
        name_map = build_reactome_name_map(reactome_path)
        ranking = pd.DataFrame({
            "pathway_id": pathway_labels,
            "pathway_name": [name_map.get(pid, pid) for pid in pathway_labels],
            "mean_abs_ig_attribution": mean_abs_attr,
        }).sort_values("mean_abs_ig_attribution", ascending=False)

        out_path = os.path.join(out_dir, "pathway_ig_rankings.csv")
        ranking.to_csv(out_path, index=False)
        logger.info(f"Saved pathway IG rankings to {out_path}")

        # Per-patient signed attributions, for follow-up analysis (e.g. do high-risk vs.
        # low-risk patients rely on different pathways? subgroup-specific attribution?)
        np.save(os.path.join(out_dir, "pathway_ig_attributions_per_patient.npy"), all_attrs)

    return pathway_labels, mean_abs_attr, all_attrs, deltas, reactome_path


# ---- Multi-seed aggregation ----

def aggregate_over_seeds(runs_dir, model_name, seeds, reactome_path_override=None,
                         n_steps=50, out_dir=None, min_mean_percentile=50.0, logger=None):
    """
    Runs pathway-level IG attribution independently for each seed of one model
    config (e.g. all 10 seeds of "full_hybinn"), then aggregates into mean ± std
    attribution per pathway across seeds -- mirroring how aggregate_results.py
    aggregates C-index across seeds, and analyze_results.py's pathway_rankings.csv.

    Each seed's own pathway_ig_rankings.csv / .npy are still written to their
    individual run_dir (via run_single_seed), so per-seed results remain
    independently inspectable; this only ADDS a combined summary on top.

    Args:
        runs_dir (str): base directory containing <model_name>/seed_<k>/ subdirs
        model_name (str): e.g. "full_hybinn", "binn_only"
        seeds (list[int]): which seeds to include
        reactome_path_override (str): override reactome path for all seeds
        n_steps (int): IG interpolation steps
        out_dir (str): where to save the aggregated CSV (default: <runs_dir>/<model_name>)
        min_mean_percentile (float): a pathway must have mean_abs_ig_attribution at or
                                      above this percentile (across ALL pathways) to be
                                      eligible for the consistency ranking. Because
                                      n_seeds is identical across every pathway,
                                      consistency_score = mean/SE = sqrt(n)/CV --
                                      i.e. it is mathematically just inverse-CV, and
                                      carries NO information about the size of the
                                      mean. Ranking by consistency_score alone (no
                                      floor) will surface pathways that are reliably
                                      SMALL, not reliably important -- the same
                                      pitfall as ranking genes by t-statistic with no
                                      fold-change floor in differential expression.
                                      Default 50.0 = require at least median raw
                                      attribution magnitude before "consistent" is a
                                      meaningful claim.
        logger: shared logger, or None to create a fresh one

    Returns:
        agg_df (DataFrame): full table, unfiltered, with both rankings and an
                             `above_mean_floor` flag indicating consistency-ranking
                             eligibility under min_mean_percentile
    """
    logger = logger or get_logger(__name__)
    out_dir = out_dir or os.path.join(runs_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    per_pathway_attrs = {}    # pathway_id -> list of per-seed mean_abs_attr
    name_map = None

    for seed in seeds:
        run_dir = os.path.join(runs_dir, model_name, f"seed_{seed}")
        if not os.path.isdir(run_dir):
            logger.info(f"[MISSING] {run_dir} -- skipping seed {seed}")
            continue

        logger.info(f"\n{'=' * 50}\nSeed {seed}\n{'=' * 50}")
        pathway_labels, mean_abs_attr, all_attrs, deltas, reactome_path = run_single_seed(
            run_dir=run_dir,
            reactome_path_override=reactome_path_override,
            n_steps=n_steps,
            save_outputs=True,   # keep per-seed CSV/npy for inspection/debugging
            logger=logger,
        )

        for pid, val in zip(pathway_labels, mean_abs_attr):
            per_pathway_attrs.setdefault(pid, []).append(val)

        if name_map is None:
            name_map = build_reactome_name_map(reactome_path)

    if not per_pathway_attrs:
        raise RuntimeError(f"No valid seed runs found under {os.path.join(runs_dir, model_name)}")

    rows = []
    for pid, vals in per_pathway_attrs.items():
        vals = np.array(vals)
        rows.append({
            "pathway_id": pid,
            "pathway_name": name_map.get(pid, pid),
            "mean_abs_ig_attribution": vals.mean(),
            "std_abs_ig_attribution": vals.std(),
            "n_seeds": len(vals),
        })

    agg_df = pd.DataFrame(rows)

    # ---- Naive ranking: by raw mean attribution ----
    agg_df = agg_df.sort_values("mean_abs_ig_attribution", ascending=False).reset_index(drop=True)
    agg_df["rank_by_mean"] = np.arange(1, len(agg_df) + 1)

    # ---- Consistency metrics ----
    agg_df["se_abs_ig_attribution"] = agg_df["std_abs_ig_attribution"] / np.sqrt(agg_df["n_seeds"])
    agg_df["cv_abs_ig_attribution"] = agg_df["std_abs_ig_attribution"] / agg_df["mean_abs_ig_attribution"]
    agg_df["consistency_score"] = agg_df["mean_abs_ig_attribution"] / agg_df["se_abs_ig_attribution"].replace(0, np.nan)

    # ---- Magnitude floor: only pathways at/above this percentile are eligible for
    # ---- the consistency ranking, so "consistent" can't just mean "consistently small" ----
    mean_floor = np.percentile(agg_df["mean_abs_ig_attribution"], min_mean_percentile)
    agg_df["above_mean_floor"] = agg_df["mean_abs_ig_attribution"] >= mean_floor

    qualified = agg_df[agg_df["above_mean_floor"]].copy()
    qualified = qualified.sort_values("consistency_score", ascending=False).reset_index(drop=True)
    qualified["rank_by_consistency_filtered"] = np.arange(1, len(qualified) + 1)

    agg_df = agg_df.merge(
        qualified[["pathway_id", "rank_by_consistency_filtered"]],
        on="pathway_id", how="left"
    )

    # ---- Unfiltered consistency ranking, kept ONLY to make the magnitude-floor's
    # ---- effect visible -- do not report this one on its own (see docstring) ----
    unfiltered_by_consistency = agg_df.sort_values("consistency_score", ascending=False)

    n_high_cv_in_top10 = (agg_df.sort_values("rank_by_mean").head(10)["cv_abs_ig_attribution"] > 1.0).sum()
    logger.info(f"\n{n_high_cv_in_top10}/10 pathways in the naive mean-ranked top 10 have CV > 1 "
               f"(std exceeds mean across seeds) -- these are outlier-seed-driven, not stable signal.")

    logger.info(f"\nMagnitude floor: pathways must be >= {min_mean_percentile:.0f}th percentile "
               f"of mean |IG| attribution ({mean_floor:.6g}) to be eligible for consistency ranking. "
               f"{len(qualified)}/{len(agg_df)} pathways qualify.")

    out_path = os.path.join(out_dir, f"pathway_ig_rankings_{model_name}_aggregated.csv")
    agg_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved aggregated pathway IG rankings ({len(agg_df)} pathways, "
               f"up to {len(seeds)} seeds) to {out_path}")

    logger.info("\nTop 10 by raw mean |IG attribution| (naive ranking):")
    logger.info(agg_df.sort_values("rank_by_mean").head(10)
               [["pathway_name", "mean_abs_ig_attribution", "std_abs_ig_attribution",
                 "cv_abs_ig_attribution"]].to_string(index=False))

    logger.info("\nTop 10 by consistency score WITHOUT magnitude floor "
               "(for comparison only -- mathematically just inverse-CV, do not report this list):")
    logger.info(unfiltered_by_consistency.head(10)
               [["pathway_name", "mean_abs_ig_attribution", "cv_abs_ig_attribution",
                 "rank_by_mean"]].to_string(index=False))

    logger.info(f"\nTop 10 by consistency score AMONG pathways above the "
               f"{min_mean_percentile:.0f}th-percentile magnitude floor (report this one):")
    logger.info(qualified.head(10)
               [["pathway_name", "mean_abs_ig_attribution", "cv_abs_ig_attribution",
                 "consistency_score", "rank_by_mean"]].to_string(index=False))

    return agg_df


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()

    # Single-run mode
    parser.add_argument('--run_dir', type=str, default=None,
                        help="Single frozen run directory (contains best_model.pt, config_frozen.yaml)")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config_frozen.yaml (default: <run_dir>/config_frozen.yaml)")

    # Multi-seed aggregation mode
    parser.add_argument('--runs_dir', type=str, default=None,
                        help="Base runs directory, e.g. experiments/runs (enables aggregation mode)")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Model config subfolder under --runs_dir, e.g. full_hybinn")
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)),
                        help="Seeds to include in aggregation mode (default: 0-9)")
    parser.add_argument('--min_mean_percentile', type=float, default=50.0,
                        help="Percentile of raw mean |IG| attribution a pathway must "
                             "clear to be eligible for consistency ranking (default: 50 = median)")

    # Shared options
    parser.add_argument('--reactome_path', type=str, default=None,
                        help="Override reactome path from config, if it moved since training")
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Where to save outputs (default: run_dir, or <runs_dir>/<model_name> in aggregate mode)")
    args = parser.parse_args()

    logger = get_logger(__name__)

    if args.runs_dir and args.model_name:
        aggregate_over_seeds(
            runs_dir=args.runs_dir,
            model_name=args.model_name,
            seeds=args.seeds,
            reactome_path_override=args.reactome_path,
            n_steps=args.n_steps,
            out_dir=args.out_dir,
            min_mean_percentile=args.min_mean_percentile,
            logger=logger,
        )
    elif args.run_dir:
        run_single_seed(
            run_dir=args.run_dir,
            config_path=args.config,
            reactome_path_override=args.reactome_path,
            n_steps=args.n_steps,
            out_dir=args.out_dir,
            save_outputs=True,
            logger=logger,
        )
    else:
        parser.error("Provide either --run_dir (single seed) or --runs_dir + --model_name (aggregate over seeds).")


if __name__ == "__main__":
    main()