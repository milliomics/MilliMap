# MilliMap: Spatial Omics Viewer

A lightweight, interactive 3D viewer for spatial omics AnnData (.h5ad) with filtering, analysis, and semi-automatic cell-type annotation.

## Overview

MilliMap is a PyQt5 + PyVista application for exploring large spatial transcriptomics datasets in 3D with real-time filtering and analysis. It includes tools for differential expression, violin plots, and a semi-automatic annotation workflow from marker genes.

## Key Features

### Visualization and Filtering
- 3D interactive scatter via PyVista (top-down camera by default)
- Color schemes: Plotly/D3, Seaborn, Custom Turbo, Milliomics, or colors from AnnData
- Multi-select filters: clusters, sections (sources), and gene expression thresholds
- Hover dialog with per-cell info; optional selection tools (polygon, circle, point)

### Expression Analytics
- Violin plots (standard and advanced) with per-gene and per-group options
- DE analysis:
  - Top DEG expression heatmap (Scanpy if available; robust fallbacks)
  - DE Volcano plot with q-value and |log2FC| thresholds

### Annotation Tools
- Cluster Annotation Helper: add/edit/remove cluster → annotation mappings; CSV load/save
- Semi-automatic Marker Annotation dialog:
  - Load markers from CSV/XLSX: columns `gene_id` (or `gene`), `annotation`, optional `weight`
  - Two modes:
    - Per-gene: score each gene’s distribution per cluster (mean↑, variance↓)
    - Per-annotation: aggregate multiple markers per annotation (co-expression aware)
  - Controls for thresholds, top-K, weights; Mean/Variance only toggle
  - Apply best labels to the in-session annotations and export tables

## Installation

### Prerequisites
```bash
# Using conda (recommended)
conda create -n millimap python=3.9
conda activate millimap
```

### Dependencies
```bash
pip install -r millimap/requirements.txt

# Optional for Excel markers/truth files
pip install openpyxl
```

## Project Structure

```
Milliomics/
├── millimap/
│   ├── Code/
│   │   ├── main.py                    # App entry
│   │   ├── viewer.py                  # Main UI and rendering
│   │   ├── analysis.py                # Analysis mode tools
│   │   ├── annotation.py              # Semi-automatic marker annotation
│   │   ├── colors.py                  # Color palettes
│   │   └── param_search_annotation.py # Parameter search script
│   ├── Icons/
│   │   └── cakeinvert.png
│   ├── README.md
│   └── requirements.txt
└── data/ …
```

## Usage

### Launch the viewer
```bash
python millimap/Code/main.py
# or as module if cwd is the package root
python -m millimap.Code.main
```

### Load data
- Drag & drop an `.h5ad` onto the window, or click “Load File”.
- Optionally “Load Gene Data” to visualize spot-level or neurite/soma features if available.

### AnnData requirements
- `obsm['spatial']`: (n_cells × 2 or 3). 2D is auto-lifted to 3D; sections can be offset along z.
- Recommended (optional): `obs['clusters']`, `obs['source']`
- Expression matrix in `.X` (or additional `layers`) used by analytics.

## Semi-automatic annotation

### Marker format
```csv
gene_id,annotation,weight
GAD1,GABAergic neuron,1.0
SLC17A7,Excitatory neuron,1.0
PVALB,PV interneuron,1.2
```
- Also supports `.xlsx`/`.xls` with the same columns; `weight` is optional.
- Matching is case-insensitive; if `var['gene_id']` exists it’s preferred.

### Modes and options
- Per-gene mode: ranks clusters for each gene using only mean and variance by default (Mean/Variance only toggle). Thresholds: per-gene min score, top-K, global min.
- Per-annotation mode: aggregates scores from multiple markers per annotation (co-expression aware). Controls for top-K, min gene score, coverage/consistency weights. Mean/Variance only sets α=1, β=γ=0.
- Apply labels: merges into in-session annotations; Load/Save via helper.

## Parameter search (offline)

Use the standalone script to search parameters against ground-truth labels:
```bash
python millimap/Code/param_search_annotation.py \
  --adata /path/to/data.h5ad \
  --markers /path/to/markers.xlsx \
  --truth /path/to/cluster_truth.csv \
  --mode both \
  --layer "X (current)" \
  --search-weights --search-ann-weights --search-topk --search-thresholds \
  --mean-var-only --max-combinations 400
```
Outputs: `<prefix>_results.csv`, `<prefix>_best_config.json`, `<prefix>_best_predictions.csv`.

## Color schemes

Palettes: Plotly/D3, Custom Turbo, Seaborn, Milliomics, or colors carried in AnnData. Switch via the “Colors” button.

## Tips
- For clustering upstream, prefer highly variable genes (HVGs) or a focused panel rather than all genes for large datasets.
- When using small targeted panels, clustering with all genes is acceptable.

## License

MIT License. See `LICENSE`.

## Support
- Open issues or questions in your repository tracker, or contact the maintainers.

—

MilliMap — fast, flexible visualization and annotation for spatial omics