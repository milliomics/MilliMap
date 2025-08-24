"""Semi-automatic cluster annotation from marker genes.

This module provides a dialog and utilities to score clusters per marker gene
based on expression distribution shape (high mean, low variance, high percent
expressed, and per-gene uniqueness across clusters). The dialog integrates with
the viewer to apply labels into the in-session cluster annotations.

Notes:
- Gene resolution is case-insensitive and prefers matching by `var['gene_id']`
  when present; falls back to `var_names`.
- Supports choosing between `X` and a named `layer`.
- Avoids densifying large matrices; for small marker lists, dense is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ------------------------------- Data structures ------------------------------

@dataclass
class MarkerSpec:
    gene: str
    annotation: str
    weight: float = 1.0


# --------------------------------- Utilities ----------------------------------

def _resolve_gene_ids(adata, raw_genes: List[str]) -> Tuple[List[str], List[str]]:
    """Resolve a list of input genes to adata.var_names.

    Returns (resolved_var_names, missing_inputs).
    Case-insensitive; includes mapping via var['gene_id'|'gene_name'] when present.
    """
    if adata is None or adata.n_vars == 0:
        return [], list(raw_genes)

    varnames = [str(v) for v in list(adata.var_names)]
    lower_map: Dict[str, str] = {v.lower(): v for v in varnames}
    # Include common id columns
    if getattr(adata, "var", None) is not None:
        for col in ("gene_id", "gene_ids", "hb_gene_id", "geneID", "gene_name", "hb_gene_name"):
            if col in adata.var.columns:
                for orig, mapped in zip(adata.var_names, adata.var[col].astype(str)):
                    lower_map[str(mapped).lower()] = str(orig)

    resolved: List[str] = []
    missing: List[str] = []
    for g in raw_genes:
        k = str(g).strip().lower()
        if not k:
            continue
        if k in lower_map:
            v = lower_map[k]
            if v not in resolved:
                resolved.append(v)
        else:
            missing.append(str(g))
    return resolved, missing


def _get_expression_matrix(adata, genes: List[str], layer: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, gene_indices) where X is (n_cells x n_genes_selected).

    Handles sparse by converting only the selected columns to dense.
    """
    if not genes:
        return np.zeros((adata.n_obs, 0)), np.array([], dtype=int)
    # Map genes to indices
    idx = np.array([int(np.where(adata.var_names == g)[0][0]) for g in genes], dtype=int)

    Xfull = adata.layers[layer] if (layer is not None and layer != "X (current)") else adata.X
    try:
        # Slice columns efficiently then densify that slice
        Xsel = Xfull[:, idx]
        if hasattr(Xsel, "toarray"):
            Xsel = Xsel.toarray()
        else:
            Xsel = np.asarray(Xsel)
    except Exception:
        # Fallback: densify whole X (not ideal but robust)
        X = Xfull.toarray() if hasattr(Xfull, "toarray") else np.asarray(Xfull)
        Xsel = X[:, idx]
    return Xsel, idx


def _compute_cluster_indices(adata, group_by: str, scope_mask: Optional[np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
    """Return (ordered_group_labels, list_of_index_arrays). Uses visible-mask scope when provided."""
    groups = adata.obs[group_by].astype(str).values
    if scope_mask is not None:
        used_groups = np.sort(np.unique(groups[scope_mask]))
    else:
        used_groups = np.sort(adata.obs[group_by].astype(str).unique())
    # Numeric-like first for natural sorting
    try:
        used_groups = sorted(used_groups, key=lambda x: (0, float(x)) if str(x).replace('.', '', 1).isdigit() else (1, str(x)))
    except Exception:
        used_groups = list(used_groups)
    idxs = [np.where(groups == g)[0] for g in used_groups]
    return list(map(str, used_groups)), idxs


def _robust_variance(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """IQR-based variance proxy; falls back to np.var on failure."""
    try:
        q75 = np.nanpercentile(X, 75, axis=axis)
        q25 = np.nanpercentile(X, 25, axis=axis)
        iqr = q75 - q25
        # For Gaussian, IQR ≈ 1.349σ => σ^2 ≈ (IQR/1.349)^2
        return (iqr / 1.349) ** 2
    except Exception:
        return np.nanvar(X, axis=axis)


def _rank01(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Return ranks in [0,1] where 1 is best. Handles ties by average rank."""
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n == 0:
        return vals
    order = np.argsort(vals)
    if higher_is_better:
        order = order  # ascending, will invert later
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    if higher_is_better:
        ranks = n + 1 - ranks  # highest value -> rank n
    # Normalize to [0,1]
    return (ranks - 1) / (n - 1) if n > 1 else np.ones_like(ranks)


def score_markers_across_clusters(
    adata,
    markers: List[MarkerSpec],
    group_by: str = "clusters",
    scope_mask: Optional[np.ndarray] = None,
    layer: Optional[str] = None,
    log1p: bool = False,
    weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05),  # mean, var_penalty, pct, uniq
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Compute scores for each (gene, cluster) and return detailed and summary tables.

    Returns:
      - per_gene_df: rows (gene, annotation, cluster, score, mean, var, pct, uniq)
      - per_cluster_best_df: rows (cluster, best_gene, best_annotation, best_score)
      - missing_genes: list of markers not found
    """
    # Resolve markers to var_names
    raw_genes = [m.gene for m in markers]
    resolved, missing = _resolve_gene_ids(adata, raw_genes)
    if not resolved:
        return pd.DataFrame(), pd.DataFrame(), missing

    # Keep only markers that resolved
    name_to_marker = {m.gene.lower(): m for m in markers}
    resolved_markers: List[MarkerSpec] = []
    for g in resolved:
        # Map back to provided annotation; if multiple matched different keys, fallback to gene as annotation
        mk = name_to_marker.get(g.lower())
        if mk is None:
            # try by inverse map via gene_id/name in var
            mk = MarkerSpec(gene=g, annotation=g, weight=1.0)
        resolved_markers.append(MarkerSpec(gene=g, annotation=mk.annotation, weight=mk.weight))

    # Build group indices
    if group_by not in adata.obs:
        raise ValueError(f"'{group_by}' not found in adata.obs")
    clusters, cluster_idxs = _compute_cluster_indices(adata, group_by, scope_mask)

    # Expression slice for selected genes
    genes = [m.gene for m in resolved_markers]
    Xsel, _ = _get_expression_matrix(adata, genes, layer)
    if log1p:
        Xsel = np.log1p(Xsel)

    # Compute per-cluster stats
    n_clusters = len(clusters)
    n_genes = len(genes)
    means = np.zeros((n_clusters, n_genes), dtype=float)
    vars_ = np.zeros_like(means)
    pcts = np.zeros_like(means)

    for i, idx in enumerate(cluster_idxs):
        if idx.size == 0:
            continue
        Xi = Xsel[idx, :]
        means[i, :] = np.asarray(Xi.mean(axis=0)).ravel()
        try:
            vars_[i, :] = _robust_variance(np.asarray(Xi), axis=0)
        except Exception:
            vars_[i, :] = np.asarray(np.var(Xi, axis=0)).ravel()
        try:
            if hasattr(Xi, "toarray"):
                Xi_ = Xi.toarray()
            else:
                Xi_ = np.asarray(Xi)
            pcts[i, :] = (Xi_ > 0).mean(axis=0)
        except Exception:
            pcts[i, :] = 0.0

    # Uniqueness: mean minus median across clusters per gene
    med_across = np.median(means, axis=0)
    uniq = means - med_across[None, :]

    w_mu, w_varpen, w_pct, w_uniq = weights

    # Build detailed per-gene scoring
    rows = []
    for j, g in enumerate(genes):
        r_mu = _rank01(means[:, j], higher_is_better=True)
        r_var = _rank01(vars_[:, j], higher_is_better=False)
        r_pct = _rank01(pcts[:, j], higher_is_better=True)
        r_uni = _rank01(uniq[:, j], higher_is_better=True)
        score = w_mu * r_mu + w_varpen * r_var + w_pct * r_pct + w_uniq * r_uni
        ann = next((m.annotation for m in resolved_markers if m.gene == g), g)
        for i, cl in enumerate(clusters):
            rows.append({
                "gene": g,
                "annotation": ann,
                "cluster": str(cl),
                "score": float(score[i]),
                "mean": float(means[i, j]),
                "var": float(vars_[i, j]),
                "pct": float(pcts[i, j]),
                "uniq": float(uniq[i, j]),
            })

    per_gene_df = pd.DataFrame(rows)

    # Per-cluster best gene
    best_rows = []
    for cl in clusters:
        sub = per_gene_df[per_gene_df["cluster"] == str(cl)]
        if len(sub) == 0:
            continue
        top = sub.sort_values(["score", "mean"], ascending=[False, False]).iloc[0]
        best_rows.append({
            "cluster": str(cl),
            "best_gene": str(top["gene"]),
            "best_annotation": str(top["annotation"]),
            "best_score": float(top["score"]),
        })
    best_df = pd.DataFrame(best_rows)

    return per_gene_df, best_df, missing


def select_assignments(
    per_gene_df: pd.DataFrame,
    top_k_per_gene: int = 2,
    per_gene_min_score: float = 0.4,
    global_min_score: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (candidates_df, final_df).

    candidates_df: rows (cluster, gene, annotation, score), filtered by per-gene rules.
    final_df: rows (cluster, label, score) where label is the best among candidates, or 'Unknown'.
    """
    if per_gene_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Per-gene selection
    cand_rows = []
    for g, gdf in per_gene_df.groupby("gene"):
        gdf_sorted = gdf.sort_values("score", ascending=False)
        gdf_sorted = gdf_sorted[gdf_sorted["score"] >= per_gene_min_score]
        if top_k_per_gene > 0:
            gdf_sorted = gdf_sorted.head(top_k_per_gene)
        for _, r in gdf_sorted.iterrows():
            cand_rows.append({
                "cluster": str(r["cluster"]),
                "gene": str(r["gene"]),
                "annotation": str(r.get("annotation", r["gene"])),
                "score": float(r["score"]),
            })
    candidates_df = pd.DataFrame(cand_rows)

    # Final by cluster
    final_rows = []
    for cl, cdf in candidates_df.groupby("cluster"):
        top = cdf.sort_values("score", ascending=False).iloc[0]
        if float(top["score"]) < global_min_score:
            final_rows.append({"cluster": str(cl), "label": "Unknown", "score": float(top["score"])})
        else:
            final_rows.append({"cluster": str(cl), "label": str(top["annotation"]), "score": float(top["score"])})
    final_df = pd.DataFrame(final_rows)
    return candidates_df, final_df


# --------------------------------- Dialog UI ----------------------------------

# -------- Annotation-level aggregation helpers --------

def _score_unique_genes_across_clusters(
    adata,
    genes: List[str],
    group_by: str,
    scope_mask: Optional[np.ndarray],
    layer: Optional[str],
    log1p: bool,
    weights: Tuple[float, float, float, float],
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return clusters, genes, score, mean, var, pct, uniq arrays with shape (n_clusters, n_genes)."""
    if group_by not in adata.obs:
        raise ValueError(f"'{group_by}' not found in adata.obs")
    clusters, cluster_idxs = _compute_cluster_indices(adata, group_by, scope_mask)

    Xsel, _ = _get_expression_matrix(adata, genes, layer)
    if log1p:
        Xsel = np.log1p(Xsel)

    n_clusters = len(clusters)
    n_genes = len(genes)
    means = np.zeros((n_clusters, n_genes), dtype=float)
    vars_ = np.zeros_like(means)
    pcts = np.zeros_like(means)

    for i, idx in enumerate(cluster_idxs):
        if idx.size == 0:
            continue
        Xi = Xsel[idx, :]
        means[i, :] = np.asarray(Xi.mean(axis=0)).ravel()
        try:
            vars_[i, :] = _robust_variance(np.asarray(Xi), axis=0)
        except Exception:
            vars_[i, :] = np.asarray(np.var(Xi, axis=0)).ravel()
        try:
            Xi_ = Xi.toarray() if hasattr(Xi, "toarray") else np.asarray(Xi)
            pcts[i, :] = (Xi_ > 0).mean(axis=0)
        except Exception:
            pcts[i, :] = 0.0

    med_across = np.median(means, axis=0)
    uniq = means - med_across[None, :]

    w_mu, w_varpen, w_pct, w_uniq = weights
    scores = np.zeros_like(means)
    for j in range(n_genes):
        r_mu = _rank01(means[:, j], higher_is_better=True)
        r_var = _rank01(vars_[:, j], higher_is_better=False)
        r_pct = _rank01(pcts[:, j], higher_is_better=True)
        r_uni = _rank01(uniq[:, j], higher_is_better=True)
        scores[:, j] = w_mu * r_mu + w_varpen * r_var + w_pct * r_pct + w_uniq * r_uni

    return clusters, genes, scores, means, vars_, pcts, uniq


def _build_marker_graph(adata, markers_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], List[str], List[str]]:
    """Build mapping annotation -> {gene(var_name): weight}.

    Returns (graph, resolved_genes_unique, missing_genes_list).
    """
    if markers_df is None or markers_df.empty:
        return {}, [], []
    genes_raw = [str(g) for g in markers_df["gene"].astype(str).tolist()]
    resolved_all, missing = _resolve_gene_ids(adata, genes_raw)
    # Build a map from input gene string to resolved var_name
    input_to_resolved: Dict[str, str] = {}
    for g in genes_raw:
        rg, _ = _resolve_gene_ids(adata, [g])
        if rg:
            input_to_resolved[g] = rg[0]

    graph: Dict[str, Dict[str, float]] = {}
    for _, row in markers_df.iterrows():
        g_in = str(row["gene"]).strip()
        ann = str(row["annotation"]).strip()
        w = float(row.get("weight", 1.0))
        vg = input_to_resolved.get(g_in)
        if vg is None:
            continue
        if ann not in graph:
            graph[ann] = {}
        graph[ann][vg] = graph[ann].get(vg, 0.0) + w
    resolved_unique = sorted({g for ad in graph.values() for g in ad.keys()})
    return graph, resolved_unique, missing


def _aggregate_annotation_scores(
    clusters: List[str],
    genes: List[str],
    scores: np.ndarray,
    means: np.ndarray,
    pcts: np.ndarray,
    marker_graph: Dict[str, Dict[str, float]],
    top_k: int = 3,
    min_gene_score: float = 0.4,
    min_pct: float = 0.0,
    min_mean: float = 0.0,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    min_contrib: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute annotation-level scores and final labels.

    Returns (detail_df, final_df).
    detail_df columns: annotation, cluster, A, coverage, cons_pen, contrib, top_markers
    final_df columns: cluster, label, score, margin
    """
    gene_to_idx = {g: j for j, g in enumerate(genes)}
    rows = []
    finals = []

    detail_list: List[Dict[str, object]] = []

    for ann, gdict in marker_graph.items():
        idx_w = [(gene_to_idx[g], w) for g, w in gdict.items() if g in gene_to_idx]
        if not idx_w:
            continue
        idxs = np.array([i for i, _ in idx_w], dtype=int)
        ws = np.array([w for _, w in idx_w], dtype=float)
        ws = ws / ws.sum() if ws.sum() > 0 else ws
        denom = max(1, len(idxs))
        for ci, cl in enumerate(clusters):
            s_vec = scores[ci, idxs]
            m_vec = means[ci, idxs]
            p_vec = pcts[ci, idxs]
            mask_inf = (s_vec >= min_gene_score) & (m_vec >= min_mean) & (p_vec >= min_pct)
            sel_s = s_vec[mask_inf]
            sel_m = m_vec[mask_inf]
            sel_idx = idxs[mask_inf]
            sel_w = ws[mask_inf] if ws.size == s_vec.size else None
            if sel_s.size == 0 or sel_s.size < min_contrib:
                A = 0.0
                cov = 0.0
                cons_pen = 0.0
                contrib = 0
                top_markers_str = ""
            else:
                order = np.argsort(sel_s)[::-1]
                if top_k > 0:
                    order = order[:top_k]
                top_scores = sel_s[order]
                if sel_w is not None and sel_w.size == sel_s.size:
                    top_weights = sel_w[order]
                    if top_weights.sum() > 0:
                        top_weights = top_weights / top_weights.sum()
                    S_bar = float(np.sum(top_scores * top_weights))
                else:
                    S_bar = float(np.mean(top_scores))
                contrib = int(sel_s.size)
                cov = float(contrib / denom)
                if sel_m.size >= 2:
                    q75 = float(np.percentile(sel_m, 75))
                    q25 = float(np.percentile(sel_m, 25))
                    iqr = q75 - q25
                    med = float(np.median(sel_m))
                    disp = iqr / (abs(med) + 1e-6)
                    disp = min(1.0, max(0.0, disp))
                    cons_pen = 1.0 - disp
                else:
                    cons_pen = 1.0
                A = float(alpha * S_bar + beta * cov + gamma * cons_pen)
                top_gene_names = [genes[g] for g in sel_idx[order]]
                top_markers_str = ", ".join(top_gene_names)

            detail_list.append({
                "annotation": ann,
                "cluster": str(cl),
                "A": A,
                "coverage": cov,
                "cons_pen": cons_pen,
                "contrib": contrib,
                "top_markers": top_markers_str,
            })

    detail_df = pd.DataFrame(detail_list)

    finals_rows: List[Dict[str, object]] = []
    if not detail_df.empty:
        for cl, sub in detail_df.groupby("cluster"):
            sub_sorted = sub.sort_values("A", ascending=False)
            best = sub_sorted.iloc[0]
            best_A = float(best["A"]) if len(sub_sorted) > 0 else 0.0
            second_A = float(sub_sorted.iloc[1]["A"]) if len(sub_sorted) > 1 else 0.0
            margin = best_A - second_A
            finals_rows.append({"cluster": str(cl), "label": str(best["annotation"]), "score": best_A, "margin": margin})
    final_df = pd.DataFrame(finals_rows)
    return detail_df, final_df

def open_semi_auto_annotation_dialog(viewer) -> None:
    """Open the semi-automatic annotation dialog bound to the given viewer."""
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
        QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QTabWidget, QTableWidget,
        QTableWidgetItem
    )
    from PyQt5.QtCore import Qt

    if viewer._adata is None or "clusters" not in viewer._adata.obs:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(viewer, "No Data", "Load data with 'clusters' in obs first.")
        return

    dlg = QDialog(viewer)
    dlg.setWindowTitle("Semi-automatic Marker Annotation")
    vbox = QVBoxLayout(dlg)

    # Row: markers CSV
    row1 = QHBoxLayout()
    row1.addWidget(QLabel("Markers CSV (gene_id,annotation[,weight]):"))
    pick_btn = QPushButton("Load…", dlg)
    row1.addWidget(pick_btn)
    vbox.addLayout(row1)

    # Matrix and transform controls
    row2 = QHBoxLayout()
    row2.addWidget(QLabel("Expression matrix:"))
    layer_combo = QComboBox(dlg)
    layer_combo.addItem("X (current)")
    try:
        if hasattr(viewer._adata, 'layers') and len(viewer._adata.layers.keys()) > 0:
            for k in viewer._adata.layers.keys():
                layer_combo.addItem(str(k))
    except Exception:
        pass
    row2.addWidget(layer_combo)
    log_chk = QCheckBox("log1p", dlg)
    log_chk.setChecked(False)
    row2.addWidget(log_chk)
    vbox.addLayout(row2)

    # Weights and thresholds
    row3 = QHBoxLayout()
    row3.addWidget(QLabel("Weights: mean"))
    w_mu = QDoubleSpinBox(dlg); w_mu.setRange(0, 2); w_mu.setSingleStep(0.05); w_mu.setValue(0.5); row3.addWidget(w_mu)
    row3.addWidget(QLabel("variance↓"))
    w_var = QDoubleSpinBox(dlg); w_var.setRange(0, 2); w_var.setSingleStep(0.05); w_var.setValue(0.3); row3.addWidget(w_var)
    row3.addWidget(QLabel("pct"))
    w_pct = QDoubleSpinBox(dlg); w_pct.setRange(0, 2); w_pct.setSingleStep(0.05); w_pct.setValue(0.15); row3.addWidget(w_pct)
    row3.addWidget(QLabel("uniq"))
    w_uni = QDoubleSpinBox(dlg); w_uni.setRange(0, 2); w_uni.setSingleStep(0.05); w_uni.setValue(0.05); row3.addWidget(w_uni)
    vbox.addLayout(row3)

    # Mean/Variance only toggle
    row3b = QHBoxLayout()
    mv_only_chk = QCheckBox("Mean/Variance only")
    mv_only_chk.setChecked(False)
    row3b.addWidget(mv_only_chk)
    mv_only_chk.setToolTip("Ignore percent-expressed and uniqueness; use only mean and variance")
    vbox.addLayout(row3b)

    row4 = QHBoxLayout()
    row4.addWidget(QLabel("Top-K per gene"))
    topk = QSpinBox(dlg); topk.setRange(0, 50); topk.setValue(2); row4.addWidget(topk)
    row4.addWidget(QLabel("Per-gene min score"))
    thr_gene = QDoubleSpinBox(dlg); thr_gene.setRange(0.0, 1.0); thr_gene.setSingleStep(0.01); thr_gene.setValue(0.4); row4.addWidget(thr_gene)
    row4.addWidget(QLabel("Global min score"))
    thr_global = QDoubleSpinBox(dlg); thr_global.setRange(0.0, 1.0); thr_global.setSingleStep(0.01); thr_global.setValue(0.3); row4.addWidget(thr_global)
    vbox.addLayout(row4)

    # Scope
    row5 = QHBoxLayout()
    row5.addWidget(QLabel("Cells:"))
    scope_combo = QComboBox(dlg)
    scope_combo.addItems(["Visible only", "All cells"]) 
    row5.addWidget(scope_combo)
    vbox.addLayout(row5)

    # Tabs for results
    tabs = QTabWidget(dlg)
    vbox.addWidget(tabs, 1)

    per_gene_table = QTableWidget(); per_gene_table.setColumnCount(8)
    per_gene_table.setHorizontalHeaderLabels(["gene","annotation","cluster","score","mean","var","pct","uniq"])
    tabs.addTab(per_gene_table, "Per marker")

    per_cluster_table = QTableWidget(); per_cluster_table.setColumnCount(4)
    per_cluster_table.setHorizontalHeaderLabels(["cluster","best_gene","best_annotation","best_score"])
    tabs.addTab(per_cluster_table, "Per cluster (best)")

    # Annotation-level tab
    per_annot_table = QTableWidget(); per_annot_table.setColumnCount(7)
    per_annot_table.setHorizontalHeaderLabels(["annotation","cluster","A","coverage","cons_pen","contrib","top_markers"])
    tabs.addTab(per_annot_table, "Per annotation")
    annot_final_table = QTableWidget(); annot_final_table.setColumnCount(4)
    annot_final_table.setHorizontalHeaderLabels(["cluster","label","score","margin"])
    tabs.addTab(annot_final_table, "Annotation labels")

    # Aggregation controls
    row6 = QHBoxLayout()
    row6.addWidget(QLabel("Mode:"))
    mode_combo = QComboBox(dlg)
    mode_combo.addItems(["Per-gene (current)", "Per-annotation (aggregate)"])
    row6.addWidget(mode_combo)
    row6.addWidget(QLabel("Top-K markers/annotation"))
    topk_ann = QSpinBox(dlg); topk_ann.setRange(0, 50); topk_ann.setValue(3); row6.addWidget(topk_ann)
    row6.addWidget(QLabel("Min gene score"))
    thr_gene_ann = QDoubleSpinBox(dlg); thr_gene_ann.setRange(0.0, 1.0); thr_gene_ann.setSingleStep(0.01); thr_gene_ann.setValue(0.4); row6.addWidget(thr_gene_ann)
    row6.addWidget(QLabel("Min pct expr"))
    thr_pct_ann = QDoubleSpinBox(dlg); thr_pct_ann.setRange(0.0, 1.0); thr_pct_ann.setSingleStep(0.01); thr_pct_ann.setValue(0.0); row6.addWidget(thr_pct_ann)
    row6.addWidget(QLabel("Min mean"))
    thr_mean_ann = QDoubleSpinBox(dlg); thr_mean_ann.setRange(0.0, 1e6); thr_mean_ann.setSingleStep(0.1); thr_mean_ann.setValue(0.0); row6.addWidget(thr_mean_ann)
    vbox.addLayout(row6)

    row7 = QHBoxLayout()
    row7.addWidget(QLabel("Annot weights: A=α⋅S̄ + β⋅coverage + γ⋅cons"))
    alpha_spin = QDoubleSpinBox(dlg); alpha_spin.setRange(0.0, 2.0); alpha_spin.setSingleStep(0.05); alpha_spin.setValue(0.6); row7.addWidget(alpha_spin)
    beta_spin = QDoubleSpinBox(dlg); beta_spin.setRange(0.0, 2.0); beta_spin.setSingleStep(0.05); beta_spin.setValue(0.3); row7.addWidget(beta_spin)
    gamma_spin = QDoubleSpinBox(dlg); gamma_spin.setRange(0.0, 2.0); gamma_spin.setSingleStep(0.05); gamma_spin.setValue(0.1); row7.addWidget(gamma_spin)
    row7.addWidget(QLabel("Min contrib"))
    min_contrib_spin = QSpinBox(dlg); min_contrib_spin.setRange(1, 50); min_contrib_spin.setValue(1); row7.addWidget(min_contrib_spin)
    vbox.addLayout(row7)

    def _apply_mv_only_state(checked: bool):
        if checked:
            # Force pct/uniq weights to 0
            try:
                w_pct.setValue(0.0); w_pct.setEnabled(False)
                w_uni.setValue(0.0); w_uni.setEnabled(False)
            except Exception:
                pass
            # Prefer expression (mean) over variance by default
            try:
                w_mu.setValue(0.7)
                w_var.setValue(0.3)
            except Exception:
                pass
            # Force annotation agg to rely only on S̄
            try:
                thr_pct_ann.setValue(0.0); thr_pct_ann.setEnabled(False)
            except Exception:
                pass
            try:
                alpha_spin.setValue(1.0); alpha_spin.setEnabled(False)
                beta_spin.setValue(0.0); beta_spin.setEnabled(False)
                gamma_spin.setValue(0.0); gamma_spin.setEnabled(False)
            except Exception:
                pass
        else:
            try:
                w_pct.setEnabled(True); w_uni.setEnabled(True)
                thr_pct_ann.setEnabled(True)
                alpha_spin.setEnabled(True); beta_spin.setEnabled(True); gamma_spin.setEnabled(True)
            except Exception:
                pass

    mv_only_chk.toggled.connect(_apply_mv_only_state)

    # Buttons
    btns = QHBoxLayout()
    run_btn = QPushButton("Run", dlg)
    apply_btn = QPushButton("Apply Best Labels", dlg)
    export_btn = QPushButton("Export…", dlg)
    close_btn = QPushButton("Close", dlg)
    btns.addStretch(); btns.addWidget(run_btn); btns.addWidget(apply_btn); btns.addWidget(export_btn); btns.addWidget(close_btn)
    vbox.addLayout(btns)

    markers_df: Optional[pd.DataFrame] = None
    per_gene_df: Optional[pd.DataFrame] = None
    best_df: Optional[pd.DataFrame] = None

    def load_markers():
        nonlocal markers_df
        fn, _ = QFileDialog.getOpenFileName(dlg, "Load Markers", "", "Data files (*.csv *.xlsx *.xls)")
        if not fn:
            return
        try:
            if fn.lower().endswith((".xlsx", ".xls")):
                try:
                    df = pd.read_excel(fn)
                except Exception as exc_x:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.critical(dlg, "Load Error", f"Failed to read Excel file. If missing, install 'openpyxl'.\nDetails: {exc_x}")
                    return
            else:
                df = pd.read_csv(fn)
            cols = {c.lower(): c for c in df.columns}
            if "gene_id" not in cols and "gene" not in cols:
                raise ValueError("Marker file must include 'gene_id' or 'gene' column")
            if "annotation" not in cols:
                raise ValueError("Marker file must include 'annotation' column")
            if "weight" not in cols:
                df["weight"] = 1.0
            # Normalize columns
            gcol = cols.get("gene_id", cols.get("gene"))
            markers_df = pd.DataFrame({
                "gene": df[gcol].astype(str).values,
                "annotation": df[cols["annotation"]].astype(str).values,
                "weight": df.get("weight", pd.Series(1.0, index=df.index)).astype(float).values,
            })
        except Exception as exc:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(dlg, "Load Error", f"Failed to load markers:\n{exc}")

    pick_btn.clicked.connect(load_markers)

    def populate_table_from_df(table: QTableWidget, df: pd.DataFrame, columns: List[str]):
        table.clearContents()
        table.setRowCount(len(df))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        for i, (_, r) in enumerate(df.iterrows()):
            for j, c in enumerate(columns):
                table.setItem(i, j, QTableWidgetItem(str(r.get(c, ""))))
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def run_scoring():
        nonlocal per_gene_df, best_df
        if markers_df is None or markers_df.empty:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(dlg, "No Markers", "Load a markers CSV first.")
            return
        # Build marker specs
        marker_list = [MarkerSpec(gene=row["gene"], annotation=row["annotation"], weight=float(row.get("weight", 1.0))) for _, row in markers_df.iterrows()]
        scope_mask = None
        if scope_combo.currentText().startswith("Visible"):
            # Use viewer's current visible mask if available
            if hasattr(viewer, "_current_mask") and isinstance(getattr(viewer, "_current_mask"), np.ndarray):
                scope_mask = viewer._current_mask
        try:
            # Always compute per-gene scores first
            per_gene_df, _best_df0, missing = score_markers_across_clusters(
                viewer._adata,
                marker_list,
                group_by="clusters",
                scope_mask=scope_mask,
                layer=layer_combo.currentText(),
                log1p=log_chk.isChecked(),
                weights=(w_mu.value(), w_var.value(), w_pct.value(), w_uni.value()),
            )
            if len(missing) > 0:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(dlg, "Missing Markers", f"Not found in dataset: {', '.join(map(str, missing))}")
            populate_table_from_df(per_gene_table, per_gene_df.sort_values(["gene","score"], ascending=[True, False]), ["gene","annotation","cluster","score","mean","var","pct","uniq"])

            mode_is_annot = (mode_combo.currentIndex() == 1)
            if not mode_is_annot:
                # Per-gene selection
                candidates_df, finals_df = select_assignments(
                    per_gene_df,
                    top_k_per_gene=int(topk.value()),
                    per_gene_min_score=float(thr_gene.value()),
                    global_min_score=float(thr_global.value()),
                )
                populate_table_from_df(per_cluster_table, finals_df.sort_values(["cluster"]), ["cluster","label","score"])
                best_df = finals_df
                # Clear annotation tables
                populate_table_from_df(per_annot_table, pd.DataFrame(), ["annotation","cluster","A","coverage","cons_pen","contrib","top_markers"])
                populate_table_from_df(annot_final_table, pd.DataFrame(), ["cluster","label","score","margin"])
            else:
                # Build marker graph (annotation -> genes)
                graph, resolved_genes, _ = _build_marker_graph(viewer._adata, pd.DataFrame(marker_list))
                # Need gene-level arrays to aggregate
                genes_unique = sorted(set([r["gene"] for _, r in pd.DataFrame(marker_list).iterrows()]))
                clusters, genes, S, means, vars_, pcts, uniq = _score_unique_genes_across_clusters(
                    viewer._adata,
                    genes_unique,
                    group_by="clusters",
                    scope_mask=scope_mask,
                    layer=layer_combo.currentText(),
                    log1p=log_chk.isChecked(),
                    weights=(w_mu.value(), w_var.value(), w_pct.value(), w_uni.value()),
                )
                detail_df, finals_df = _aggregate_annotation_scores(
                    clusters,
                    genes,
                    S,
                    means,
                    pcts,
                    graph,
                    top_k=int(topk_ann.value()),
                    min_gene_score=float(thr_gene_ann.value()),
                    min_pct=float(thr_pct_ann.value()),
                    min_mean=float(thr_mean_ann.value()),
                    alpha=float(alpha_spin.value()),
                    beta=float(beta_spin.value()),
                    gamma=float(gamma_spin.value()),
                    min_contrib=int(min_contrib_spin.value()),
                )
                populate_table_from_df(per_annot_table, detail_df.sort_values(["annotation","A"], ascending=[True, False]), ["annotation","cluster","A","coverage","cons_pen","contrib","top_markers"])
                populate_table_from_df(annot_final_table, finals_df.sort_values(["cluster"]), ["cluster","label","score","margin"])
                best_df = finals_df
        except Exception as exc:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(dlg, "Scoring Error", f"Failed to score markers:\n{exc}")

    def apply_best_labels():
        if best_df is None or best_df.empty:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(dlg, "No Results", "Run scoring first.")
            return
        # Merge into viewer annotations (skip 'Unknown')
        updated = 0
        for _, r in best_df.iterrows():
            cl = str(r.get("cluster", ""))
            lab = str(r.get("label", ""))
            if lab and lab.lower() != "unknown":
                viewer._cluster_annotations[cl] = lab
                updated += 1
        try:
            viewer._refresh_cluster_list_labels()
        except Exception:
            pass
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(dlg, "Applied", f"Applied {updated} labels to cluster annotations.")

    def export_results():
        if per_gene_df is None or per_gene_df.empty:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(dlg, "No Results", "Run scoring first.")
            return
        fn, _ = QFileDialog.getSaveFileName(dlg, "Save Results (prefix)", "marker_annotation", "CSV files (*.csv)")
        if not fn:
            return
        try:
            per_gene_df.to_csv(fn if fn.endswith(".csv") else fn + "_per_marker.csv", index=False)
            if best_df is not None and not best_df.empty:
                # Normalize to cluster,label,score
                out = best_df.copy()
                if "best_annotation" in out.columns:
                    out.rename(columns={"best_annotation":"label","best_score":"score"}, inplace=True)
                out.to_csv((fn if fn.endswith(".csv") else fn + "_final.csv"), index=False)
        except Exception as exc:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(dlg, "Save Error", f"Failed to save results:\n{exc}")

    run_btn.clicked.connect(run_scoring)
    apply_btn.clicked.connect(apply_best_labels)
    export_btn.clicked.connect(export_results)
    close_btn.clicked.connect(dlg.accept)

    dlg.resize(900, 600)
    dlg.exec_()


