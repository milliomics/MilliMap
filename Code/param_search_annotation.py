"""Parameter search for semi-automatic marker-based cluster annotation.

Usage (example):
  python param_search_annotation.py \
    --adata /path/to/data.h5ad \
    --markers /path/to/markers.csv \
    --truth /path/to/cluster_truth.csv \
    --layer "X (current)" \
    --mode both \
    --max-combinations 200

Inputs:
  - AnnData with obs['clusters'] (cluster ids as strings/numbers)
  - Markers file: CSV/XLSX with columns: gene_id(or gene), annotation[, weight]
  - Truth file (cluster-level preferred): CSV with columns: cluster, label
    (Optionally, cell-level truth: CSV with columns: cell, label)

Outputs:
  - results CSV with metrics for each parameter set
  - best predictions CSV for the best-scoring configuration

Notes:
  - Avoids plotting; purely numeric search.
  - Uses functions from annotation.py (per-gene scoring and annotation aggregation).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import anndata as ad

# Robust import like other modules
try:
    from .annotation import (
        score_markers_across_clusters,
        select_assignments,
        _build_marker_graph,
        _score_unique_genes_across_clusters,
        _resolve_gene_ids,
        _aggregate_annotation_scores,
    )
except Exception:
    from annotation import (
        score_markers_across_clusters,
        select_assignments,
        _build_marker_graph,
        _score_unique_genes_across_clusters,
        _resolve_gene_ids,
        _aggregate_annotation_scores,
    )


def read_markers(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "gene_id" not in cols and "gene" not in cols:
        raise ValueError("Marker file must include 'gene_id' or 'gene' column")
    if "annotation" not in cols:
        raise ValueError("Marker file must include 'annotation' column")
    if "weight" not in cols:
        df["weight"] = 1.0
    gcol = cols.get("gene_id", cols.get("gene"))
    out = pd.DataFrame({
        "gene": df[gcol].astype(str).values,
        "annotation": df[cols["annotation"]].astype(str).values,
        "weight": df.get("weight", pd.Series(1.0, index=df.index)).astype(float).values,
    })
    return out


def read_truth(path: str) -> Tuple[str, pd.DataFrame]:
    """Return (mode, df). mode in {"cluster", "cell"}."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "cluster" in cols and "label" in cols:
        return "cluster", pd.DataFrame({
            "cluster": df[cols["cluster"]].astype(str),
            "label": df[cols["label"]].astype(str)
        })
    if "cell" in cols and "label" in cols:
        return "cell", pd.DataFrame({
            "cell": df[cols["cell"]].astype(str),
            "label": df[cols["label"]].astype(str)
        })
    raise ValueError("Truth file must have columns ('cluster','label') or ('cell','label')")


def evaluate_cluster_labels(
    adata: ad.AnnData,
    pred_cluster_to_label: Dict[str, str],
    truth_mode: str,
    truth_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute accuracy metrics at cluster level or cell level.

    Returns dict with metrics: acc, acc_excl_unknown, coverage, macro_f1 (labels in truth), n_eval.
    """
    from collections import Counter
    # Prepare vectors of y_true, y_pred for evaluation
    y_true: List[str] = []
    y_pred: List[str] = []

    if truth_mode == "cluster":
        for _, r in truth_df.iterrows():
            cl = str(r["cluster"])
            lab = str(r["label"])
            if cl in pred_cluster_to_label:
                y_true.append(lab)
                y_pred.append(pred_cluster_to_label[cl])
    else:  # cell-level
        obs_names_to_cluster = adata.obs["clusters"].astype(str)
        truth_map = {str(r["cell"]): str(r["label"]) for _, r in truth_df.iterrows()}
        for cell, lab in truth_map.items():
            if cell in adata.obs_names:
                cl = str(obs_names_to_cluster.loc[cell])
                pred_lab = pred_cluster_to_label.get(cl, "Unknown")
                y_true.append(lab)
                y_pred.append(pred_lab)

    if len(y_true) == 0:
        return {"acc": 0.0, "acc_excl_unknown": 0.0, "coverage": 0.0, "macro_f1": 0.0, "n_eval": 0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    mask_known = y_pred != "Unknown"
    acc_ex = float(np.mean((y_true[mask_known] == y_pred[mask_known])) if mask_known.any() else 0.0)
    coverage = float(np.mean(mask_known))
    # Macro F1 over labels present in truth
    labels = sorted(set(y_true))
    f1s = []
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"acc": acc, "acc_excl_unknown": acc_ex, "coverage": coverage, "macro_f1": macro_f1, "n_eval": len(y_true)}


def generate_param_grid(args) -> List[Dict[str, object]]:
    """Generate parameter grid dictionaries. Honors --mode."""
    grids: List[Dict[str, object]] = []

    # common
    log1p_opts = [False, True] if args.search_log1p else [args.log1p]
    layers = [args.layer]

    if args.mode in ("both", "gene"):
        if args.search_weights:
            if args.mean_var_only:
                gene_weight_sets = [
                    (0.7, 0.3, 0.0, 0.0),
                    (0.6, 0.4, 0.0, 0.0),
                    (0.5, 0.5, 0.0, 0.0),
                ]
            else:
                gene_weight_sets = [
                    (0.6, 0.2, 0.15, 0.05),
                    (0.5, 0.3, 0.15, 0.05),
                    (0.4, 0.4, 0.15, 0.05),
                ]
        else:
            gene_weight_sets = [(
                args.w_mu,
                args.w_var,
                0.0 if args.mean_var_only else args.w_pct,
                0.0 if args.mean_var_only else args.w_uniq,
            )]
        topk_gene = [1, 2, 3] if args.search_topk else [args.topk_gene]
        min_gene = [0.3, 0.4, 0.5] if args.search_thresholds else [args.min_gene]
        min_global = [0.3, 0.4] if args.search_thresholds else [args.min_global]
        for layer, log1p, wset, tk, mg, mglo in itertools.product(layers, log1p_opts, gene_weight_sets, topk_gene, min_gene, min_global):
            grids.append({
                "mode": "gene",
                "layer": layer,
                "log1p": log1p,
                "w_mu": wset[0], "w_var": wset[1], "w_pct": wset[2], "w_uniq": wset[3],
                "topk_gene": int(tk), "min_gene": float(mg), "min_global": float(mglo),
            })

    if args.mode in ("both", "annot"):
        if args.search_ann_weights:
            if args.mean_var_only:
                annot_weight_sets = [(1.0, 0.0, 0.0)]
            else:
                annot_weight_sets = [
                    (0.6, 0.3, 0.1),
                    (0.7, 0.2, 0.1),
                ]
        else:
            annot_weight_sets = [(
                1.0 if args.mean_var_only else args.alpha,
                0.0 if args.mean_var_only else args.beta,
                0.0 if args.mean_var_only else args.gamma,
            )]
        topk_ann = [2, 3, 5] if args.search_topk else [args.topk_ann]
        min_gene_ann = [0.3, 0.4, 0.5] if args.search_thresholds else [args.min_gene_ann]
        min_pct = [0.0] if args.mean_var_only else ([0.0, 0.1] if args.search_thresholds else [args.min_pct])
        min_mean = [0.0] if not args.search_thresholds else [0.0]
        min_contrib = [1, 2] if args.search_thresholds else [args.min_contrib]
        for layer, log1p, wset, tk, mg, mp, mm, mc in itertools.product(layers, log1p_opts, annot_weight_sets, topk_ann, min_gene_ann, min_pct, min_mean, min_contrib):
            grids.append({
                "mode": "annot",
                "layer": layer,
                "log1p": log1p,
                "alpha": wset[0], "beta": wset[1], "gamma": wset[2],
                "topk_ann": int(tk), "min_gene_ann": float(mg), "min_pct": float(mp), "min_mean": float(mm), "min_contrib": int(mc),
                # per-gene weights still control S components
                "w_mu": args.w_mu, "w_var": args.w_var, "w_pct": args.w_pct, "w_uniq": args.w_uniq,
            })

    # Optionally sample to limit combinations
    if args.max_combinations and len(grids) > args.max_combinations:
        random.seed(args.seed)
        grids = random.sample(grids, args.max_combinations)
    return grids


def run_param_search(args) -> None:
    adata = ad.read_h5ad(args.adata)
    if "clusters" not in adata.obs:
        raise ValueError("AnnData must contain obs['clusters']")
    markers_df = read_markers(args.markers)
    truth_mode, truth_df = read_truth(args.truth)

    results: List[Dict[str, object]] = []
    best_score = -1.0
    best_conf: Optional[Dict[str, object]] = None
    best_pred_map: Optional[Dict[str, str]] = None

    param_grid = generate_param_grid(args)
    # Progress bar if available
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(param_grid, total=len(param_grid), desc="Param search")
    except Exception:
        iterator = param_grid

    for i, conf in enumerate(iterator, start=1):
        mode = conf["mode"]
        try:
            if mode == "gene":
                # Per-gene scoring and selection
                per_gene_df, _best_df0, missing = score_markers_across_clusters(
                    adata,
                    [
                        # Lightweight adapter
                        type("MS", (), {"gene": r["gene"], "annotation": r["annotation"], "weight": float(r.get("weight", 1.0))})
                        for _, r in markers_df.iterrows()
                    ],
                    group_by="clusters",
                    scope_mask=None,
                    layer=conf["layer"],
                    log1p=conf["log1p"],
                    weights=(conf["w_mu"], conf["w_var"], conf["w_pct"], conf["w_uniq"]),
                )
                candidates_df, finals_df = select_assignments(
                    per_gene_df,
                    top_k_per_gene=conf["topk_gene"],
                    per_gene_min_score=conf["min_gene"],
                    global_min_score=conf["min_global"],
                )
                pred_map = {str(r["cluster"]): str(r["label"]) for _, r in finals_df.iterrows()}
            else:
                # Annotation-level aggregation
                graph, resolved_genes, _missing = _build_marker_graph(adata, markers_df)
                clusters, genes, S, means, vars_, pcts, uniq = _score_unique_genes_across_clusters(
                    adata,
                    resolved_genes,
                    group_by="clusters",
                    scope_mask=None,
                    layer=conf["layer"],
                    log1p=conf["log1p"],
                    weights=(conf["w_mu"], conf["w_var"], conf["w_pct"], conf["w_uniq"]),
                )
                detail_df, finals_df = _aggregate_annotation_scores(
                    clusters,
                    genes,
                    S,
                    means,
                    pcts,
                    graph,
                    top_k=conf["topk_ann"],
                    min_gene_score=conf["min_gene_ann"],
                    min_pct=conf["min_pct"],
                    min_mean=conf["min_mean"],
                    alpha=conf["alpha"],
                    beta=conf["beta"],
                    gamma=conf["gamma"],
                    min_contrib=conf["min_contrib"],
                )
                pred_map = {str(r["cluster"]): str(r["label"]) for _, r in finals_df.iterrows()}

            metrics = evaluate_cluster_labels(adata, pred_map, truth_mode, truth_df)
            score = metrics.get("macro_f1", 0.0)
            rec = {**conf, **metrics}
            results.append(rec)
            if score > best_score:
                best_score = score
                best_conf = conf.copy()
                best_pred_map = pred_map.copy()
        except Exception as exc:
            rec = {**conf, "error": str(exc)}
            results.append(rec)
        # Fallback textual progress if tqdm is not available
        if 'tqdm' not in globals():
            if i == 1 or i == len(param_grid) or i % max(1, len(param_grid)//10) == 0:
                print(f"Progress: {i}/{len(param_grid)}")

    res_df = pd.DataFrame(results)
    out_prefix = args.out_prefix or os.path.splitext(os.path.basename(args.adata))[0] + "_paramsearch"
    res_path = out_prefix + "_results.csv"
    res_df.to_csv(res_path, index=False)

    if best_conf is not None and best_pred_map is not None:
        best_path = out_prefix + "_best_predictions.csv"
        pd.DataFrame({"cluster": list(best_pred_map.keys()), "label": list(best_pred_map.values())}).to_csv(best_path, index=False)
        conf_path = out_prefix + "_best_config.json"
        with open(conf_path, "w") as f:
            json.dump(best_conf, f, indent=2)

    print(f"Wrote results to: {res_path}")
    if best_conf is not None:
        print(f"Best macro-F1: {best_score:.3f}")
        print(f"Best config JSON: {out_prefix + '_best_config.json'}")
        print(f"Best predictions CSV: {out_prefix + '_best_predictions.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parameter search for marker-based cluster annotation")
    p.add_argument("--adata", required=True, help="Path to .h5ad file")
    p.add_argument("--markers", required=True, help="Path to markers CSV/XLSX")
    p.add_argument("--truth", required=True, help="Path to truth CSV (cluster,label) or (cell,label)")
    p.add_argument("--layer", default="X (current)", help="Expression matrix: 'X (current)' or layer name")
    p.add_argument("--mode", choices=["gene", "annot", "both"], default="both", help="Which scoring mode to search")
    p.add_argument("--out-prefix", default=None, help="Output prefix for results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-combinations", type=int, default=200, help="Cap total combinations; random sample if exceeded")

    # Base per-gene weights and options (also used within annotation mode for S)
    p.add_argument("--w-mu", type=float, default=0.5)
    p.add_argument("--w-var", type=float, default=0.3)
    p.add_argument("--w-pct", type=float, default=0.15)
    p.add_argument("--w-uniq", type=float, default=0.05)
    p.add_argument("--log1p", action="store_true")
    p.add_argument("--search-log1p", action="store_true")
    p.add_argument("--search-weights", action="store_true")
    p.add_argument("--search-topk", action="store_true")
    p.add_argument("--search-thresholds", action="store_true")

    # Gene-mode thresholds
    p.add_argument("--topk-gene", type=int, default=2)
    p.add_argument("--min-gene", type=float, default=0.4)
    p.add_argument("--min-global", type=float, default=0.3)

    # Annotation aggregation params
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--topk-ann", type=int, default=3)
    p.add_argument("--min-gene-ann", type=float, default=0.4)
    p.add_argument("--min-pct", type=float, default=0.0)
    p.add_argument("--min-mean", type=float, default=0.0)
    p.add_argument("--min-contrib", type=int, default=1)
    p.add_argument("--search-ann-weights", action="store_true")
    p.add_argument("--mean-var-only", action="store_true", help="Use only mean and variance (disable pct/uniq; set alpha=1,beta=gamma=0 in annotation mode)")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_param_search(args)


if __name__ == "__main__":
    main()


