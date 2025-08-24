"""Main spatial omics viewer application."""

import pathlib
from typing import Optional

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PyQt5.QtCore import Qt, QMimeData, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.spatial import cKDTree
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QCursor, QIcon, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QGroupBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
    QDialog,
    QScrollArea,
    QInputDialog,
    QCheckBox,
    QComboBox,
    QRadioButton,
    QFormLayout,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
)
from matplotlib.figure import Figure
from pyvistaqt import QtInteractor
import pyvista as pv
import vtk
import pandas as pd

# Handle both direct execution and package import
try:
    from .colors import (
        generate_plotly_extended_palette,
        generate_custom_turbo_palette,
        generate_sns_palette,
        generate_milliomics_palette
    )
    from .dialogs import (
        ColorSchemeDialog,
        HoverInfoDialog,
        SectionSelectDialog,
        LassoSelectionDialog
    )
    from .analysis import create_analysis_tools
    from .annotation import open_semi_auto_annotation_dialog
except (ImportError, ValueError):
    from colors import (
        generate_plotly_extended_palette,
        generate_custom_turbo_palette,
        generate_sns_palette,
        generate_milliomics_palette
    )
    from dialogs import (
        ColorSchemeDialog,
        HoverInfoDialog,
        SectionSelectDialog,
        LassoSelectionDialog
    )
    from analysis import create_analysis_tools
    from annotation import open_semi_auto_annotation_dialog


class MillimapViewer(QMainWindow):
    """Millimap: A lightweight viewer for spatial omics AnnData objects (.h5ad).

    Users can drag-and-drop an AnnData file onto the window or use the
    "Load File" button to pick a file. The viewer looks for the
    ``obsm['spatial']`` matrix (\(n\_cells \times 3\)) plus optional
    ``obs['clusters']`` categorical information to colour points.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Millimap - Spatial Omics Viewer")
        self.resize(1400, 800)
        self.setAcceptDrops(True)

        # -----------------------------------------------------------------
        # Set window icon
        # -----------------------------------------------------------------
        # Use relative path to icon file
        current_dir = pathlib.Path(__file__).parent
        icon_path = current_dir.parent / "Icons" / "cakeinvert.png"
        if icon_path.exists():
            pix = QPixmap(str(icon_path))
            # Composite over white background to avoid transparency issues
            white_bg = QPixmap(pix.size())
            white_bg.fill(Qt.white)
            painter = QPainter(white_bg)
            painter.drawPixmap(0, 0, pix)
            painter.end()
            self.setWindowIcon(QIcon(white_bg))

        # Initialize analysis mode flag and tools early
        self._analysis_mode = False
        self._analysis_tools = create_analysis_tools(self)

        # Central widget --------------------------------------------------------
        self._central_widget = QWidget(self)
        self.setCentralWidget(self._central_widget)

        # Layouts ---------------------------------------------------------------
        self._outer_layout = QVBoxLayout(self._central_widget)

        # --- Content (plot + side panel) -------------------------------------
        content_layout = QHBoxLayout()
        self._outer_layout.addLayout(content_layout, stretch=1)

        # PyVista plotter --------------------------------------------------------
        self._plotter_widget = QtInteractor(self._central_widget)
        content_layout.addWidget(self._plotter_widget.interactor, stretch=1)

        # Side panel inside a scroll area ---------------------------------------
        self._scroll_area = QScrollArea(self._central_widget)
        self._scroll_area.setWidgetResizable(True)
        self._side_panel = QWidget()
        self._side_layout = QVBoxLayout(self._side_panel)
        self._scroll_area.setWidget(self._side_panel)
        self._scroll_area.setFixedWidth(270)
        content_layout.addWidget(self._scroll_area)

        # ----------------------------- CLUSTERS -----------------------------
        self._cluster_group = QGroupBox("Clusters")
        self._cluster_group.setCheckable(True); self._cluster_group.setChecked(True)
        self._cluster_layout = QVBoxLayout(self._cluster_group)
        # Now that layout exists, connect toggle
        self._cluster_group.toggled.connect(lambda val, grp=self._cluster_layout: grp.parentWidget().setVisible(val))

        # Search bar
        cluster_search_row = QHBoxLayout()
        self._cluster_search_line = QLineEdit()
        self._cluster_search_line.setPlaceholderText("search…")
        cluster_search_btn = QPushButton("Search")
        cluster_search_btn.clicked.connect(self._perform_cluster_search)
        cluster_search_row.addWidget(self._cluster_search_line, stretch=1)
        cluster_search_row.addWidget(cluster_search_btn)
        self._cluster_layout.addLayout(cluster_search_row)

        # List
        self._cluster_list = QListWidget()
        self._cluster_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._cluster_list.itemSelectionChanged.connect(self._update_filters)
        self._cluster_layout.addWidget(self._cluster_list)

        # Clear-selection button
        clear_clusters_btn = QPushButton("Unselect Clusters")
        clear_clusters_btn.clicked.connect(self._cluster_list.clearSelection)
        self._cluster_layout.addWidget(clear_clusters_btn)

        # Cluster Annotation Helper button
        self._cluster_anno_btn = QPushButton("Cluster Annotation Helper")
        self._cluster_anno_btn.clicked.connect(self._open_cluster_annotation_helper)
        self._cluster_layout.addWidget(self._cluster_anno_btn)

        # Semi-automatic annotation button
        self._cluster_auto_btn = QPushButton("Semi-auto Annotate…")
        self._cluster_auto_btn.clicked.connect(lambda: open_semi_auto_annotation_dialog(self))
        self._cluster_layout.addWidget(self._cluster_auto_btn)

        # Top DEG expression analysis heatmap button
        self._de_heatmap_btn = QPushButton("Top DEG Expression Heatmap")
        self._de_heatmap_btn.clicked.connect(self._show_de_heatmap)
        self._cluster_layout.addWidget(self._de_heatmap_btn)

        # Differential Expression (Volcano) analysis button
        self._de_volcano_btn = QPushButton("DE Analysis (Volcano Plot)")
        self._de_volcano_btn.clicked.connect(self._show_de_volcano_dialog)
        self._cluster_layout.addWidget(self._de_volcano_btn)

        self._side_layout.addWidget(self._cluster_group)

        # ----------------------------- SECTIONS -----------------------------
        self._section_group = QGroupBox("Sections (source)")
        self._section_group.setCheckable(True); self._section_group.setChecked(True)
        self._section_layout = QVBoxLayout(self._section_group)
        self._section_group.toggled.connect(lambda val, grp=self._section_layout: grp.parentWidget().setVisible(val))

        section_search_row = QHBoxLayout()
        self._section_search_line = QLineEdit()
        self._section_search_line.setPlaceholderText("search…")
        section_search_btn = QPushButton("Search")
        section_search_btn.clicked.connect(self._perform_section_search)
        section_search_row.addWidget(self._section_search_line, stretch=1)
        section_search_row.addWidget(section_search_btn)
        self._section_layout.addLayout(section_search_row)

        # Section list and navigation buttons side-by-side ----------------
        self._section_list = QListWidget()
        self._section_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._section_list.itemSelectionChanged.connect(self._update_filters)

        up_btn = QPushButton("Up")
        up_btn.clicked.connect(self._section_up)
        down_btn = QPushButton("Down")
        down_btn.clicked.connect(self._section_down)

        nav_vlayout = QVBoxLayout()
        nav_vlayout.setSpacing(2)
        nav_vlayout.addWidget(up_btn)
        nav_vlayout.addWidget(down_btn)
        nav_vlayout.addStretch()

        list_nav_hlayout = QHBoxLayout()
        list_nav_hlayout.addWidget(self._section_list, 1)
        list_nav_hlayout.addLayout(nav_vlayout)
        self._section_layout.addLayout(list_nav_hlayout)

        clear_sections_btn = QPushButton("Unselect Sections")
        clear_sections_btn.clicked.connect(self._section_list.clearSelection)
        self._section_layout.addWidget(clear_sections_btn)

        self._side_layout.addWidget(self._section_group)

        # ------------------------------ GENES ------------------------------
        self._gene_group = QGroupBox("Genes (ALL selected; expr filter)")
        self._gene_group.setCheckable(True); self._gene_group.setChecked(True)
        self._gene_layout = QVBoxLayout(self._gene_group)
        self._gene_group.toggled.connect(lambda val, grp=self._gene_layout: grp.parentWidget().setVisible(val))

        gene_search_row = QHBoxLayout()
        self._gene_search_line = QLineEdit()
        self._gene_search_line.setPlaceholderText("search…")
        gene_search_btn = QPushButton("Search")
        gene_search_btn.clicked.connect(self._perform_gene_search)
        gene_search_row.addWidget(self._gene_search_line, stretch=1)
        gene_search_row.addWidget(gene_search_btn)
        self._gene_layout.addLayout(gene_search_row)

        self._gene_list = QListWidget()
        self._gene_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._gene_list.itemSelectionChanged.connect(self._update_filters)
        self._gene_layout.addWidget(self._gene_list)

        # Clear-selection button for genes
        clear_genes_btn = QPushButton("Unselect Genes")
        clear_genes_btn.clicked.connect(self._gene_list.clearSelection)
        self._gene_layout.addWidget(clear_genes_btn)

        # Gene Only toggle button --------------------------------------
        self._gene_only_btn = QPushButton("Gene Only", self._gene_group)
        self._gene_only_btn.setCheckable(True)
        self._gene_only_btn.toggled.connect(self._toggle_gene_only_mode)
        self._gene_layout.addWidget(self._gene_only_btn)

        # Expression threshold
        self._expr_threshold = QDoubleSpinBox()
        self._expr_threshold.setRange(0, 1e6)
        self._expr_threshold.setValue(0)
        self._expr_threshold.setSingleStep(1)
        self._expr_threshold.setPrefix("expr > ")
        self._expr_threshold.valueChanged.connect(self._update_filters)
        self._gene_layout.addWidget(self._expr_threshold)

        # Violin plot button for selected genes across visible clusters
        self._violin_btn = QPushButton("Violin Plot", self._gene_group)
        self._violin_btn.clicked.connect(self._show_violin_plot)
        self._gene_layout.addWidget(self._violin_btn)

        # Advanced violin plot button with flexible options
        self._violin_adv_btn = QPushButton("Violin (Advanced)", self._gene_group)
        self._violin_adv_btn.clicked.connect(self._show_violin_advanced)
        self._gene_layout.addWidget(self._violin_adv_btn)

        self._side_layout.addWidget(self._gene_group)

        # ----------------------- SELECTION SUMMARY -----------------------
        self._summary_group = QGroupBox("Selection Summary")
        self._summary_group.setCheckable(True); self._summary_group.setChecked(True)
        self._summary_layout = QVBoxLayout(self._summary_group)
        self._summary_label = QLabel("No data loaded.")
        self._summary_label.setWordWrap(True)
        self._summary_layout.addWidget(self._summary_label)
        self._side_layout.addWidget(self._summary_group)

        self._side_layout.addStretch()

        # ----------------------------- ANALYSIS -----------------------------
        self._analysis_group = QGroupBox("Analysis Tools")
        self._analysis_group.setCheckable(True)
        self._analysis_group.setChecked(True)
        # When collapsed, we still keep visibility per analysis mode
        def _ana_toggle(val):
            if self._analysis_mode:
                self._analysis_group.setVisible(True)
                self._analysis_layout.parentWidget().setVisible(val)
        self._analysis_group.toggled.connect(_ana_toggle)
        self._analysis_layout = QVBoxLayout(self._analysis_group)

        ana_poly_btn = QPushButton("+ Polygon")
        ana_poly_btn.clicked.connect(self._analysis_tools.start_polygon_selection)
        ana_circle_btn = QPushButton("+ Circle")
        ana_circle_btn.clicked.connect(self._analysis_tools.start_circle_selection)
        ana_point_btn = QPushButton("+ Point")
        ana_point_btn.clicked.connect(self._analysis_tools.start_point_selection)

        self._analysis_layout.addWidget(ana_poly_btn)
        self._analysis_layout.addWidget(ana_circle_btn)
        self._analysis_layout.addWidget(ana_point_btn)
        self._analysis_layout.addStretch()

        self._side_layout.addWidget(self._analysis_group)

        # Control panel (bottom) -----------------------------------------------
        bottom_layout = QHBoxLayout()
        self._outer_layout.addLayout(bottom_layout)

        self._status_label = QLabel("Drop an *.h5ad file or click \"Load File\"…")
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_layout.addWidget(self._status_label, stretch=1)

        self._colors_btn = QPushButton("Colors", self)
        self._colors_btn.clicked.connect(self._open_color_dialog)
        bottom_layout.addWidget(self._colors_btn)

        # Analysis mode toggle button
        self._analysis_btn = QPushButton("Analysis Mode", self)
        self._analysis_btn.setCheckable(True)
        self._analysis_btn.toggled.connect(self._analysis_tools.toggle_analysis_mode)
        bottom_layout.addWidget(self._analysis_btn)

        self._load_btn = QPushButton("Load File", self)
        self._load_btn.clicked.connect(self._open_file_dialog)
        bottom_layout.addWidget(self._load_btn)

        # Button to load Gene AnnData (spot-level)
        self._load_gene_btn = QPushButton("Load Gene Data", self)
        self._load_gene_btn.clicked.connect(self._open_gene_adata_dialog)
        bottom_layout.addWidget(self._load_gene_btn)

        # Mode flag
        self._mode = "cell"  # "gene" (cell-based spot csv) or "gene_spots"
        self._gene_only = False  # toggle state



        # Default opacity for rendered selected points (0.5 = 50% transparency)
        self._point_opacity = 0.6
        
        # Color scheme selection
        self._color_scheme = "plotly_d3"  # "anndata", "plotly_d3", "custom_turbo", "sns_palette", "milliomics"

        # Gene spot AnnData
        self._gene_adata: Optional[ad.AnnData] = None

        # Internal state ---------------------------------------------------------
        self._adata: Optional[ad.AnnData] = None
        self._current_clusters: list[str] = []
        self._current_sources: list[str] = []
        self._current_genes: list[str] = []

        # Camera handling
        self._has_initial_camera = False
        self._orientation_added = False

        # Hover/highlight state -------------------------------------------
        self._highlight_actor = None
        self._hover_idx = -1
        self._coords_all: Optional[np.ndarray] = None
        self._hover_dialog = HoverInfoDialog(self)

        # New state for precise point picking
        self._full_pick_actor = None

        # New state for precise point picking
        self._picker = vtk.vtkPointPicker()
        self._picker.SetTolerance(0.01)
        self._plotter_widget.interactor.SetPicker(self._picker)
        self._plotter_widget.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)

        # -------------------------------- Analysis helpers --------------------
        self._analysis_selection_actor = None
        self._kd_tree: Optional[cKDTree] = None

        # In-session memory for cluster annotations: {cluster_label(str): annotation(str)}
        self._cluster_annotations: dict[str, str] = {}

    # ---------------------------------------------------------------------
    # Cluster Annotation Helper
    # ---------------------------------------------------------------------
    def _open_cluster_annotation_helper(self) -> None:
        if self._adata is None or "clusters" not in self._adata.obs:
            QMessageBox.information(self, "No Clusters", "Load data with 'clusters' in obs first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Cluster Annotation Helper")
        layout = QVBoxLayout(dlg)

        # Inputs row: clusters and name
        row = QHBoxLayout()
        clusters_edit = QLineEdit(dlg)
        clusters_edit.setPlaceholderText("Cluster numbers (comma-separated), e.g., 1,2,5")
        name_edit = QLineEdit(dlg)
        name_edit.setPlaceholderText("Annotation (e.g., Excitatory neuron)")
        add_btn = QPushButton("Add", dlg)
        row.addWidget(clusters_edit, 2)
        row.addWidget(name_edit, 2)
        row.addWidget(add_btn, 0)
        layout.addLayout(row)

        # Current annotations list
        from PyQt5.QtWidgets import QListWidget
        anno_list = QListWidget(dlg)
        def refresh_list():
            anno_list.clear()
            for cl, nm in sorted(self._cluster_annotations.items(), key=lambda kv: kv[0]):
                anno_list.addItem(f"{cl} → {nm}")
        refresh_list()
        layout.addWidget(anno_list)

        # Remove, Load and Save buttons
        btns = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected", dlg)
        load_btn = QPushButton("Load…", dlg)
        save_btn = QPushButton("Save…", dlg)
        close_btn = QPushButton("Close", dlg)
        btns.addWidget(remove_btn)
        btns.addWidget(load_btn)
        btns.addStretch()
        btns.addWidget(save_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        def on_add():
            clusters_text = clusters_edit.text().strip()
            name_text = name_edit.text().strip()
            if not clusters_text or not name_text:
                QMessageBox.information(dlg, "Missing Input", "Provide clusters and an annotation name.")
                return
            # Normalize cluster labels as strings present in obs
            valid_clusters = set(self._adata.obs["clusters"].astype(str).unique())
            items = [c.strip() for c in clusters_text.replace(";", ",").split(",") if c.strip()]
            not_found = [c for c in items if c not in valid_clusters]
            if not_found:
                QMessageBox.information(dlg, "Unknown Clusters", f"These clusters are not in data: {', '.join(not_found)}")
                return
            for c in items:
                self._cluster_annotations[c] = name_text
            clusters_edit.clear(); name_edit.clear()
            refresh_list()
            # Also refresh cluster list labels to show brackets
            try:
                self._refresh_cluster_list_labels()
            except Exception:
                pass

        def on_remove():
            sel = anno_list.selectedItems()
            if not sel:
                return
            for it in sel:
                txt = it.text()
                cl = txt.split("→")[0].strip()
                if cl in self._cluster_annotations:
                    del self._cluster_annotations[cl]
            refresh_list()
            # Also refresh cluster list labels to remove brackets
            try:
                self._refresh_cluster_list_labels()
            except Exception:
                pass

        def on_save():
            if not self._cluster_annotations:
                QMessageBox.information(dlg, "Nothing to Save", "No annotations defined yet.")
                return
            fname, _ = QFileDialog.getSaveFileName(dlg, "Save Cluster Annotations", "cluster_annotations.csv", "CSV files (*.csv)")
            if not fname:
                return
            try:
                import csv
                with open(fname, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["cluster", "annotation"]) 
                    for cl, nm in sorted(self._cluster_annotations.items(), key=lambda kv: kv[0]):
                        w.writerow([cl, nm])
                QMessageBox.information(dlg, "Saved", f"Annotations saved to:\n{fname}")
            except Exception as exc:
                QMessageBox.critical(dlg, "Save Error", f"Failed to save annotations:\n{exc}")

        add_btn.clicked.connect(on_add)
        remove_btn.clicked.connect(on_remove)
        
        def on_load():
            fname, _ = QFileDialog.getOpenFileName(dlg, "Load Cluster Annotations", "", "CSV files (*.csv)")
            if not fname:
                return
            try:
                import csv
                with open(fname, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    cols = {c.lower(): c for c in (reader.fieldnames or [])}
                    if "cluster" not in cols or "annotation" not in cols:
                        QMessageBox.information(dlg, "Invalid File", "CSV must have 'cluster' and 'annotation' columns.")
                        return
                    loaded = {}
                    for row in reader:
                        cl = str(row[cols["cluster"]]).strip()
                        nm = str(row[cols["annotation"]]).strip()
                        if cl:
                            loaded[cl] = nm
                # Validate clusters exist in current data
                try:
                    valid_clusters = set(self._adata.obs["clusters"].astype(str).unique())
                except Exception:
                    valid_clusters = None
                unknown = []
                for cl, nm in loaded.items():
                    if valid_clusters is not None and cl not in valid_clusters:
                        unknown.append(cl)
                        continue
                    self._cluster_annotations[cl] = nm
                refresh_list()
                try:
                    self._refresh_cluster_list_labels()
                except Exception:
                    pass
                if unknown:
                    QMessageBox.information(dlg, "Partial Load", f"Loaded annotations, but these clusters were not found and were skipped: {', '.join(unknown)}")
                else:
                    QMessageBox.information(dlg, "Loaded", "Cluster annotations loaded.")
            except Exception as exc:
                QMessageBox.critical(dlg, "Load Error", f"Failed to load annotations:\n{exc}")

        load_btn.clicked.connect(on_load)
        save_btn.clicked.connect(on_save)
        close_btn.clicked.connect(dlg.accept)

        dlg.resize(600, 400)
        dlg.exec_()

    # ---------------------------------------------------------------------
    # Differential Expression Heatmap (Scanpy)
    # ---------------------------------------------------------------------
    def _show_de_heatmap(self) -> None:
        if self._adata is None or "clusters" not in self._adata.obs:
            QMessageBox.information(self, "No Clusters", "Load data with 'clusters' in obs first.")
            return
        selected_clusters = [itm.text() for itm in self._cluster_list.selectedItems()]
        # If none selected in the list, treat as selecting all currently visible clusters;
        # if no filters are active, this becomes all clusters in the dataset.
        if not selected_clusters:
            clusters_series_all = self._adata.obs["clusters"].astype(str)
            if hasattr(self, "_current_mask") and isinstance(self._current_mask, np.ndarray) and self._current_mask.any():
                selected_clusters = list(np.sort(clusters_series_all[self._current_mask].unique()))
            else:
                selected_clusters = list(np.sort(clusters_series_all.unique()))

        # Allow manual override of clusters via input dialog
        try:
            suggestion = ", ".join(selected_clusters)
            text, ok = QInputDialog.getText(self, "Choose Clusters",
                                            "Clusters to include (comma-separated).\nLeave empty to use current selection:",
                                            text=suggestion)
            if ok and text.strip():
                cand = [c.strip() for c in text.replace(";", ",").split(",") if c.strip()]
                # Validate
                valid = set(self._adata.obs["clusters"].astype(str).unique())
                selected_clusters = [c for c in cand if c in valid]
        except Exception:
            pass
        if len(selected_clusters) < 2:
            QMessageBox.information(self, "Select Clusters", "Need at least two clusters for DE analysis.")
            return

        # Determine selected sources (sections). If none selected, use all.
        selected_sources = [itm.text() for itm in self._section_list.selectedItems()]
        use_sources_filter = bool(selected_sources) and ("source" in self._adata.obs)

        # Ask for top-N genes
        n_top, ok = QInputDialog.getInt(self, "Top DE genes", "Number of top genes per cluster:", 50, 1, 2000, 1)
        if not ok:
            return

        # Prepare data
        has_scanpy = True
        try:
            import scanpy as sc  # optional, preferred path
        except Exception:
            has_scanpy = False

        try:
            clusters_series = self._adata.obs["clusters"].astype(str)
            mask = clusters_series.isin(selected_clusters).values
            if use_sources_filter:
                src_mask = self._adata.obs["source"].astype(str).isin(selected_sources).values
                mask &= src_mask
            if not mask.any():
                QMessageBox.information(self, "No Cells", "No cells found for the selected clusters.")
                return
            adata_sub = self._adata[mask].copy()
            # Ensure groupby column is string dtype and enforce numeric cluster ordering
            adata_sub.obs["clusters"] = adata_sub.obs["clusters"].astype(str)
            def _num_key(x):
                try:
                    return (0, float(x))
                except Exception:
                    return (1, str(x))
            ordered_all = sorted(list(adata_sub.obs["clusters"].unique()), key=_num_key)
            if selected_clusters:
                selset = set(selected_clusters)
                ordered_all = [c for c in ordered_all if c in selset]
            adata_sub.obs["clusters"] = pd.Categorical(adata_sub.obs["clusters"], categories=ordered_all, ordered=True)

            # Use module-level imports; avoid redeclaring numpy/pandas locally

            # Ask user which matrix to use (X or a layer)
            selected_layer = None
            try:
                items = ["X (current)"]
                if hasattr(self._adata, 'layers') and len(self._adata.layers.keys()) > 0:
                    items += [str(k) for k in self._adata.layers.keys()]
                choice, ok = QInputDialog.getItem(self, "Expression Matrix", "Use matrix:", items, 0, False)
                if ok and choice != "X (current)":
                    selected_layer = choice
            except Exception:
                selected_layer = None

            # Determine gene set
            ordered_genes: list[str] = []
            if has_scanpy:
                try:
                    # Prefer scanpy layer parameter; fallback to replacing X when unsupported
                    if selected_layer is not None:
                        try:
                            sc.tl.rank_genes_groups(adata_sub, groupby="clusters", method="wilcoxon", n_genes=n_top, layer=selected_layer)
                            adata_rank = adata_sub
                        except TypeError:
                            adata_tmp = adata_sub.copy(); adata_tmp.X = adata_sub.layers[selected_layer]
                            sc.tl.rank_genes_groups(adata_tmp, groupby="clusters", method="wilcoxon", n_genes=n_top)
                            adata_rank = adata_tmp
                    else:
                        sc.tl.rank_genes_groups(adata_sub, groupby="clusters", method="wilcoxon", n_genes=n_top)
                        adata_rank = adata_sub
                    gene_list: list[str] = []
                    for grp in sorted(adata_sub.obs["clusters"].astype(str).unique()):
                        try:
                            df_grp = sc.get.rank_genes_groups_df(adata_rank, group=grp)
                            gene_list.extend(df_grp["names"].head(n_top).tolist())
                        except Exception:
                            pass
                    seen = set()
                    for g in gene_list:
                        if g not in seen:
                            seen.add(g); ordered_genes.append(g)
                except Exception:
                    has_scanpy = False

            if not has_scanpy:
                # Fallback: pick top genes by max mean difference across clusters
                try:
                    X_full = adata_sub.layers[selected_layer] if selected_layer is not None else adata_sub.X
                    if hasattr(X_full, "toarray"):
                        # avoid full densify; compute per-cluster means on sparse directly
                        pass  # csr.mean handles sparse, so no conversion
                    clusters_vals = adata_sub.obs["clusters"].astype(str).values
                    unique_cls = [c for c in selected_clusters if c in set(clusters_vals)]
                    # Compute cluster means vectorized per cluster
                    means_per_cluster = []
                    for c in unique_cls:
                        idxs = np.where(clusters_vals == c)[0]
                        if idxs.size == 0:
                            continue
                        m = X_full[idxs].mean(axis=0)
                        m = np.asarray(m).ravel()
                        means_per_cluster.append(m)
                    if not means_per_cluster:
                        QMessageBox.information(self, "No DE Genes", "No data available for selected clusters.")
                        return
                    means_mat = np.vstack(means_per_cluster)  # shape: n_clusters x n_genes
                    # Replace NaNs/Infs before calculations
                    means_mat = np.nan_to_num(means_mat, nan=0.0, posinf=0.0, neginf=0.0)
                    # Score genes by max pairwise difference across clusters
                    max_diff = means_mat.max(axis=0) - means_mat.min(axis=0)
                    # Get top indices
                    top_idx = np.argsort(-max_diff)[:n_top]
                    ordered_genes = [str(g) for g in adata_sub.var_names[top_idx]]
                except Exception as exc:
                    QMessageBox.critical(self, "DE Fallback Error", f"Failed to compute fallback genes:\n{exc}")
                    return

            if not ordered_genes:
                QMessageBox.information(self, "No DE Genes", "Could not compute DE genes.")
                return

            # Determine color cap (vmax) to avoid saturation
            cap_percentile = 99.0
            vmax_val = None
            try:
                cap_percentile, ok = QInputDialog.getDouble(
                    self,
                    "Color Cap (Percentile)",
                    "Upper expression cap percentile (0-100). Values above this will be clipped:",
                    99.0, 50.0, 100.0, 1
                )
                if not ok:
                    cap_percentile = 99.0
            except Exception:
                cap_percentile = 99.0
            try:
                if selected_layer is None:
                    X_for_cap = adata_sub[:, ordered_genes].X
                else:
                    idxs_cap = [int(np.where(adata_sub.var_names == g)[0][0]) for g in ordered_genes if g in adata_sub.var_names]
                    X_for_cap = adata_sub.layers[selected_layer][:, idxs_cap]
                if hasattr(X_for_cap, 'toarray'):
                    X_for_cap = X_for_cap.toarray()
                X_for_cap = np.nan_to_num(X_for_cap, nan=0.0, posinf=0.0, neginf=0.0)
                vmax_val = float(np.percentile(X_for_cap, cap_percentile)) if X_for_cap.size > 0 else None
                if vmax_val is not None and vmax_val <= 0:
                    vmax_val = None
            except Exception:
                vmax_val = None

            # Try Scanpy heatmap first if available
            fig = None
            if has_scanpy:
                try:
                    # Try enabling dendrogram to cluster rows so similar clusters are adjacent
                    g = sc.pl.heatmap(
                        adata_sub,
                        var_names=ordered_genes,
                        groupby="clusters",
                        show=False,
                        dendrogram=True,
                        cmap='viridis',
                        vmin=0,
                        vmax=vmax_val if 'vmax_val' in locals() and vmax_val is not None else None
                    )
                    fig = g.fig if hasattr(g, 'fig') else plt.gcf()
                except Exception:
                    try:
                        # Fallback without dendrogram if unavailable
                        g = sc.pl.heatmap(
                            adata_sub,
                            var_names=ordered_genes,
                            groupby="clusters",
                            show=False,
                            dendrogram=False,
                            cmap='viridis',
                            vmin=0,
                            vmax=vmax_val if 'vmax_val' in locals() and vmax_val is not None else None
                        )
                        fig = g.fig if hasattr(g, 'fig') else plt.gcf()
                    except Exception:
                        fig = None
            
            if fig is None:
                # Fallback: seaborn heatmap of mean expression per cluster
                import seaborn as sns
                if selected_layer is None:
                    X = adata_sub[:, ordered_genes].X
                else:
                    idxs = [int(np.where(adata_sub.var_names == g)[0][0]) for g in ordered_genes if g in adata_sub.var_names]
                    X = adata_sub.layers[selected_layer][:, idxs]
                if hasattr(X, "toarray"):
                    X = X.toarray()
                # Clean any NaN/Inf which can lead to NaN log2FC downstream
                try:
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
                df_expr = pd.DataFrame(X, columns=ordered_genes)
                df_expr['cluster'] = adata_sub.obs['clusters'].astype(str).values
                mean_expr = df_expr.groupby('cluster')[ordered_genes].mean()
                # Reorder clusters (rows) and genes (columns) by hierarchical clustering of expression profiles
                try:
                    from scipy.cluster.hierarchy import linkage, leaves_list
                    from scipy.spatial.distance import pdist
                    # Restrict to selected clusters if provided
                    if selected_clusters:
                        mean_expr = mean_expr.loc[[c for c in mean_expr.index if c in set(selected_clusters)]]
                    # Row order (clusters)
                    try:
                        dist = pdist(mean_expr.values, metric='correlation')
                        Z = linkage(dist, method='average')
                    except Exception:
                        # Fallback to euclidean if correlation fails (e.g., constant rows)
                        Z = linkage(mean_expr.values, method='average', metric='euclidean')
                    order_rows = leaves_list(Z)
                    mean_expr = mean_expr.iloc[order_rows]

                    # Column order (genes)
                    try:
                        dist_cols = pdist(mean_expr.values.T, metric='correlation')
                        Zc = linkage(dist_cols, method='average')
                        order_cols = leaves_list(Zc)
                        mean_expr = mean_expr.iloc[:, order_cols]
                    except Exception:
                        pass
                except Exception:
                    # Fallback: keep numeric-like ordering
                    def _num_key2(x):
                        try:
                            return (0, float(x))
                        except Exception:
                            return (1, str(x))
                    idx_list = list(mean_expr.index)
                    if selected_clusters:
                        idx_list = [c for c in idx_list if c in set(selected_clusters)]
                    mean_expr = mean_expr.loc[sorted(idx_list, key=_num_key2)]

                # Clip any negative values to zero to keep colorbar non-negative
                try:
                    mean_expr = mean_expr.clip(lower=0)
                except Exception:
                    pass

                # Determine cluster count after ordering for sizing
                _cluster_count_for_size = mean_expr.shape[0]
                fig = Figure(figsize=(max(8, len(ordered_genes)*0.3), max(4, _cluster_count_for_size*0.3 + 2)))
                ax = fig.add_subplot(111)
                # Use vmax if computed
                _vmax = vmax_val if 'vmax_val' in locals() and vmax_val is not None else None
                sns.heatmap(mean_expr, ax=ax, cmap='viridis', vmin=0, vmax=_vmax)
                # Horizontal y tick labels to avoid overlap
                try:
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                except Exception:
                    pass
                ax.set_xlabel('Gene')
                ax.set_ylabel('Cluster')
                ax.set_title(f"Mean expression (top {n_top})")

            # Show in dialog with Save + Summary buttons (+ include details option)
            dlg = QDialog(self)
            dlg.setWindowTitle("Top DEG Heatmap")
            vbox = QVBoxLayout(dlg)
            canvas = FigureCanvas(fig)
            vbox.addWidget(canvas)
            btn_row = QHBoxLayout()
            include_info_chk = QCheckBox("Include details on save", dlg)
            include_info_chk.setChecked(True)
            btn_row.addWidget(include_info_chk)
            btn_row.addStretch()
            save_btn = QPushButton("Save", dlg)
            summary_btn = QPushButton("Summary", dlg)
            close_btn = QPushButton("Close", dlg)
            btn_row.addWidget(save_btn); btn_row.addWidget(summary_btn); btn_row.addWidget(close_btn)
            vbox.addLayout(btn_row)

            def _save_heatmap():
                fname, _ = QFileDialog.getSaveFileName(dlg, "Save Heatmap", "de_heatmap.png", "PNG files (*.png)")
                if fname:
                    try:
                        # Determine target figure size based on genes/clusters to avoid squeezing
                        try:
                            num_genes = len(ordered_genes)
                            num_clusters = int(adata_sub.obs['clusters'].astype(str).nunique())
                            base_w = max(8.0, num_genes * 0.30)
                            base_h = max(4.0, num_clusters * 0.30 + 2.0)
                            cur_w, cur_h = fig.get_size_inches()
                            # Ensure at least the base size
                            target_w = max(cur_w, base_w)
                            target_h = max(cur_h, base_h)
                            # If including details, allocate extra height instead of shrinking heatmap
                            extra_h = 0.0
                            if include_info_chk.isChecked():
                                extra_h = 0.9  # room for details box
                            fig.set_size_inches(target_w, target_h + extra_h, forward=True)
                            # Keep a reasonable bottom margin for x label and details
                            try:
                                fig.subplots_adjust(bottom=0.18)
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Optionally annotate with details centered below the x-axis label
                        if include_info_chk.isChecked():
                            try:
                                sources_info = ", ".join(selected_sources) if use_sources_filter else "All sources"
                                clusters_info = ", ".join(selected_clusters)
                                n_cells = int(adata_sub.n_obs)
                                method_info = "scanpy.wilcoxon" if has_scanpy and fig is not None else "mean-expression fallback"
                                details = [
                                    f"Clusters: {clusters_info}",
                                    f"Sources: {sources_info}",
                                    f"Cells used: {n_cells:,}",
                                    f"Top genes per cluster: {n_top}",
                                    f"Method: {method_info}",
                                ]
                                # Centered further below x-axis label
                                fig.text(0.5, 0.000, "\n".join(details), ha='center', va='bottom', fontsize=9,
                                         bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
                                canvas.draw_idle()
                            except Exception:
                                pass
                        fig.savefig(fname, dpi=300, bbox_inches='tight')
                        QMessageBox.information(dlg, "Saved", f"Saved to:\n{fname}")
                    except Exception as exc:
                        QMessageBox.critical(dlg, "Save Error", f"Failed to save heatmap:\n{exc}")
            save_btn.clicked.connect(_save_heatmap)
            
            def _open_deg_summary():
                dlg2 = QDialog(self)
                dlg2.setWindowTitle("Top DEG Summary")
                lay = QVBoxLayout(dlg2)
                info = QLabel(
                    f"Clusters compared: {', '.join(selected_clusters)}\n"
                    f"Sources: {', '.join(selected_sources) if use_sources_filter else 'All sources'}\n"
                    f"Top N per cluster: {n_top}"
                )
                info.setWordWrap(True)
                lay.addWidget(info)

                tabs = QTabWidget(dlg2)
                lay.addWidget(tabs, 1)

                def make_table(df_show):
                    tbl = QTableWidget()
                    tbl.setColumnCount(len(df_show.columns))
                    tbl.setHorizontalHeaderLabels([str(c) for c in df_show.columns])
                    tbl.setRowCount(len(df_show))
                    for i, (_, r) in enumerate(df_show.iterrows()):
                        for j, c in enumerate(df_show.columns):
                            tbl.setItem(i, j, QTableWidgetItem(str(r[c])))
                    tbl.setSortingEnabled(True)
                    tbl.resizeColumnsToContents()
                    return tbl

                if has_scanpy:
                    try:
                        for grp in sorted(adata_sub.obs["clusters"].astype(str).unique()):
                            try:
                                df_grp = sc.get.rank_genes_groups_df(adata_sub, group=grp).copy()
                                # Standardize columns for display
                                rename_map = {"names":"gene", "pvals_adj":"qval", "pvals":"pval", "logfoldchanges":"log2FC"}
                                for k, v in rename_map.items():
                                    if k in df_grp.columns and v not in df_grp.columns:
                                        df_grp[v] = df_grp[k]
                                cols = [c for c in ["gene","log2FC","qval","scores"] if c in df_grp.columns]
                                df_grp = df_grp[cols].head(n_top)
                                tabs.addTab(make_table(df_grp), f"Cluster {grp}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                else:
                    # Fallback: build simple per-cluster stats for ordered_genes
                    try:
                        Xfull = adata_sub[:, ordered_genes].X
                        if hasattr(Xfull, 'toarray'):
                            Xfull = Xfull.toarray()
                        cl_series = adata_sub.obs['clusters'].astype(str).values
                        for grp in sorted(set(cl_series)):
                            idx = np.where(cl_series == grp)[0]
                            rest_idx = np.where(cl_series != grp)[0]
                            if idx.size == 0 or rest_idx.size == 0:
                                continue
                            mean_grp = Xfull[idx].mean(axis=0)
                            mean_rest = Xfull[rest_idx].mean(axis=0)
                            eps = 1e-8
                            l2fc = np.log2((mean_grp+eps)/(mean_rest+eps))
                            df_grp = pd.DataFrame({"gene": ordered_genes, "log2FC": np.asarray(l2fc).ravel()})
                            df_grp = df_grp.sort_values("log2FC", ascending=False).head(n_top)
                            tabs.addTab(make_table(df_grp), f"Cluster {grp}")
                    except Exception:
                        pass

                # Export button
                btns = QHBoxLayout()
                export_btn = QPushButton("Save Tables", dlg2)
                close2 = QPushButton("Close", dlg2)
                btns.addStretch(); btns.addWidget(export_btn); btns.addWidget(close2)
                lay.addLayout(btns)

                def _export_all():
                    fname, _ = QFileDialog.getSaveFileName(dlg2, "Save DEG Tables", "top_deg_summary.csv", "CSV files (*.csv)")
                    if not fname:
                        return
                    try:
                        # Concatenate all tabs into one CSV with a cluster column
                        import io
                        import csv
                        with open(fname, 'w', newline='') as f:
                            w = csv.writer(f)
                            w.writerow(["cluster","gene","log2FC","qval","scores"])  
                            # Recompute using same path as display for robustness
                            if has_scanpy:
                                for grp in sorted(adata_sub.obs["clusters"].astype(str).unique()):
                                    try:
                                        df_grp = sc.get.rank_genes_groups_df(adata_sub, group=grp).copy()
                                        rename_map = {"names":"gene", "pvals_adj":"qval", "pvals":"pval", "logfoldchanges":"log2FC"}
                                        for k, v in rename_map.items():
                                            if k in df_grp.columns and v not in df_grp.columns:
                                                df_grp[v] = df_grp[k]
                                        cols = [c for c in ["gene","log2FC","qval","scores"] if c in df_grp.columns]
                                        df_grp = df_grp[cols].head(n_top)
                                        for _, r in df_grp.iterrows():
                                            w.writerow([grp, r.get('gene',''), r.get('log2FC',''), r.get('qval',''), r.get('scores','')])
                                    except Exception:
                                        pass
                            else:
                                Xfull = adata_sub[:, ordered_genes].X
                                if hasattr(Xfull, 'toarray'):
                                    Xfull = Xfull.toarray()
                                cl_series = adata_sub.obs['clusters'].astype(str).values
                                for grp in sorted(set(cl_series)):
                                    idx = np.where(cl_series == grp)[0]
                                    rest_idx = np.where(cl_series != grp)[0]
                                    if idx.size == 0 or rest_idx.size == 0:
                                        continue
                                    mean_grp = Xfull[idx].mean(axis=0)
                                    mean_rest = Xfull[rest_idx].mean(axis=0)
                                    eps = 1e-8
                                    l2fc = np.log2((mean_grp+eps)/(mean_rest+eps))
                                    df_grp = pd.DataFrame({"gene": ordered_genes, "log2FC": np.asarray(l2fc).ravel()})
                                    df_grp = df_grp.sort_values("log2FC", ascending=False).head(n_top)
                                    for _, r in df_grp.iterrows():
                                        w.writerow([grp, r.get('gene',''), r.get('log2FC',''), '', ''])
                        QMessageBox.information(dlg2, "Saved", f"Tables saved to:\n{fname}")
                    except Exception as exc:
                        QMessageBox.critical(dlg2, "Save Error", f"Failed to save tables:\n{exc}")

                export_btn.clicked.connect(_export_all)
                close2.clicked.connect(dlg2.accept)
                dlg2.resize(900, 600)
                dlg2.exec_()

            summary_btn.clicked.connect(_open_deg_summary)
            close_btn.clicked.connect(dlg.accept)
            dlg.resize(1000, 700)
            dlg.exec_()

        except Exception as exc:
            QMessageBox.critical(self, "DE Analysis Error", f"Failed to compute or plot DE genes:\n{exc}")

    def _show_violin_plot(self) -> None:
        """Show violin plot of selected genes across clusters currently shown.
        Uses all cells from the clusters that are currently visible in the viewer.
        """
        if self._adata is None:
            QMessageBox.information(self, "No Data", "Load an AnnData file first.")
            return
        if "clusters" not in self._adata.obs:
            QMessageBox.information(self, "Missing Clusters", "AnnData lacks 'clusters' in obs.")
            return

        selected_genes = [itm.text() for itm in self._gene_list.selectedItems()]
        if not selected_genes:
            QMessageBox.information(self, "No Genes Selected", "Select at least one gene.")
            return

        # Determine clusters that are currently shown
        visible_mask = None
        if hasattr(self, "_current_mask") and isinstance(getattr(self, "_current_mask"), np.ndarray):
            visible_mask = self._current_mask
        else:
            # Recreate mask from current filters (similar to summary)
            mask = np.ones(self._adata.n_obs, dtype=bool)
            if self._current_clusters and "clusters" in self._adata.obs:
                mask &= self._adata.obs["clusters"].astype(str).isin(self._current_clusters).values
            if self._current_sources and "source" in self._adata.obs:
                mask &= self._adata.obs["source"].astype(str).isin(self._current_sources).values
            # Apply gene threshold only when in cell mode (matches what is shown)
            if self._mode == "cell" and self._current_genes:
                try:
                    X_sub = self._adata[:, self._current_genes].X
                    if hasattr(X_sub, "toarray"):
                        X_sub = X_sub.toarray()
                    mask &= (X_sub > float(self._expr_threshold.value())).all(axis=1)
                except Exception:
                    pass
            visible_mask = mask

        if visible_mask is None or not visible_mask.any():
            QMessageBox.information(self, "No Cells Visible", "No cells are currently shown.")
            return

        clusters_series = self._adata.obs["clusters"].astype(str)
        visible_clusters = np.sort(clusters_series[visible_mask].unique())
        if len(visible_clusters) == 0:
            QMessageBox.information(self, "No Clusters", "No clusters are currently shown.")
            return

        # Build mask including all cells from the visible clusters (ignoring other filters)
        use_mask = clusters_series.isin(visible_clusters).values

        # Validate genes exist
        missing = [g for g in selected_genes if g not in set(self._adata.var_names)]
        if missing:
            QMessageBox.information(self, "Missing Genes", f"These genes are not in the dataset: {', '.join(missing)}")
            return

        # Collect data into a DataFrame
        try:
            X = self._adata[:, selected_genes].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            import pandas as pd
            data_frames = []
            for j, gene in enumerate(selected_genes):
                expr = X[:, j]
                df = pd.DataFrame({
                    "cluster": clusters_series.values,
                    "expression": expr,
                    "gene": gene
                })
                data_frames.append(df[use_mask])
            df_all = pd.concat(data_frames, ignore_index=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to prepare data for violin plot:\n{exc}")
            return

        # Create dialog with matplotlib canvas
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Violin Plot - Gene Expression by Cluster")
            vbox = QVBoxLayout(dialog)
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            vbox.addWidget(canvas)

            ax_list = []
            try:
                import seaborn as sns  # optional, nicer plots
                if len(selected_genes) == 1:
                    ax = fig.add_subplot(111)
                    sns.violinplot(data=df_all, x="cluster", y="expression", ax=ax, cut=0, inner="quartile", scale="width")
                    ax.set_title(f"{selected_genes[0]} expression across clusters ({use_mask.sum():,} cells)")
                    ax_list.append(ax)
                else:
                    # Facet per gene to avoid overcrowding hues
                    ncols = min(3, len(selected_genes))
                    nrows = int(np.ceil(len(selected_genes) / ncols))
                    axes = fig.subplots(nrows, ncols, squeeze=False)
                    for i, gene in enumerate(selected_genes):
                        r, c = divmod(i, ncols)
                        ax = axes[r][c]
                        sns.violinplot(data=df_all[df_all["gene"] == gene], x="cluster", y="expression", ax=ax, cut=0, inner="quartile", scale="width")
                        ax.set_title(gene)
                        ax_list.append(ax)
                    # Hide any unused axes
                    for k in range(len(selected_genes), nrows * ncols):
                        r, c = divmod(k, ncols)
                        axes[r][c].axis('off')
            except Exception:
                # Fallback to pure matplotlib
                ax = fig.add_subplot(111)
                clusters_sorted = sorted(visible_clusters, key=lambda x: (False, float(x)) if str(x).replace('.', '', 1).isdigit() else (True, str(x)))
                for i, gene in enumerate(selected_genes):
                    ydata = [df_all[(df_all["cluster"] == cl) & (df_all["gene"] == gene)]["expression"].values for cl in clusters_sorted]
                    parts = ax.violinplot(ydata, positions=np.arange(len(clusters_sorted)) + i*0.1, widths=0.8/len(selected_genes), showmeans=False, showextrema=False, showmedians=True)
                ax.set_xticks(np.arange(len(clusters_sorted)))
                ax.set_xticklabels(clusters_sorted, rotation=45, ha='right')
                ax.set_title(f"Expression across clusters ({', '.join(selected_genes)})")
                ax_list.append(ax)

            # Common formatting
            for ax in ax_list:
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Expression")
                ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()

            # Summary checkbox + buttons row
            btn_row = QHBoxLayout()
            include_summary_chk = QCheckBox("Include summary at bottom", dialog)
            include_summary_chk.setChecked(False)
            btn_row.addWidget(include_summary_chk)

            btn_row.addStretch()

            save_btn = QPushButton("Save", dialog)
            def _do_save():
                # Optionally render a summary text at the bottom before saving
                if include_summary_chk.isChecked():
                    try:
                        summary_lines = [
                            f"Visible clusters: {', '.join(map(str, visible_clusters))}",
                            f"Genes: {', '.join(selected_genes)}",
                            f"Cells used: {int(use_mask.sum()):,}",
                        ]
                        # Add a bottom text box occupying full figure width
                        fig.subplots_adjust(bottom=0.22)
                        fig.text(0.01, 0.02, "\n".join(summary_lines), ha='left', va='bottom', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
                        canvas.draw_idle()
                    except Exception:
                        pass

                fname, _ = QFileDialog.getSaveFileName(dialog, "Save Violin Plot", "violin_plot.png", "PNG files (*.png)")
                if fname:
                    try:
                        fig.savefig(fname, dpi=300, bbox_inches='tight')
                        QMessageBox.information(dialog, "Saved", f"Saved to:\n{fname}")
                    except Exception as exc:
                        QMessageBox.critical(dialog, "Save Error", f"Failed to save plot:\n{exc}")
            save_btn.clicked.connect(_do_save)
            btn_row.addWidget(save_btn)

            close_btn = QPushButton("Close", dialog)
            close_btn.clicked.connect(dialog.accept)
            btn_row.addWidget(close_btn)

            vbox.addLayout(btn_row)

            dialog.resize(1000, 600)
            dialog.exec_()
        except Exception as exc:
            QMessageBox.critical(self, "Plot Error", f"Failed to draw violin plot:\n{exc}")

    def _show_violin_advanced(self) -> None:
        """Open an advanced violin plotting dialog.
        Options:
          - Group by: clusters or sources
          - Cell scope: visible cells only, or all cells from selected groups
          - Genes: selected from list, custom list, or top-N highly expressed
        """
        if self._adata is None:
            QMessageBox.information(self, "No Data", "Load an AnnData file first.")
            return

        import numpy as np
        import pandas as pd
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
            QLineEdit, QSpinBox, QCheckBox, QPushButton, QFileDialog
        )

        # Build dialog UI
        dlg = QDialog(self)
        dlg.setWindowTitle("Violin Plot (Advanced)")
        vbox = QVBoxLayout(dlg)

        # Group-by selection
        gb_row = QHBoxLayout()
        gb_row.addWidget(QLabel("Group by:"))
        gb_combo = QComboBox(dlg)
        has_clusters = "clusters" in getattr(self._adata, "obs", {})
        has_sources = "source" in getattr(self._adata, "obs", {})
        if has_clusters:
            gb_combo.addItem("clusters")
        if has_sources:
            gb_combo.addItem("source")
        if gb_combo.count() == 0:
            QMessageBox.information(self, "No Groups", "Data has no 'clusters' or 'source' in obs.")
            return
        gb_row.addWidget(gb_combo)
        vbox.addLayout(gb_row)

        # Cell scope selection
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Cells:"))
        scope_combo = QComboBox(dlg)
        scope_combo.addItems(["Visible cells only", "All cells from selected groups"]) 
        scope_row.addWidget(scope_combo)
        vbox.addLayout(scope_row)

        # Gene source selection
        gene_row = QHBoxLayout()
        gene_row.addWidget(QLabel("Genes:"))
        gene_combo = QComboBox(dlg)
        gene_combo.addItems(["Selected in list", "Enter genes…", "Top-N highly expressed"])
        gene_row.addWidget(gene_combo)

        # Inputs that are conditionally shown
        genes_edit = QLineEdit(dlg)
        genes_edit.setPlaceholderText("Comma or space separated gene IDs")
        genes_edit.setVisible(False)
        gene_row.addWidget(genes_edit, 1)

        topn_spin = QSpinBox(dlg)
        topn_spin.setRange(1, max(10, int(getattr(self._adata, 'n_vars', 100))))
        topn_spin.setValue(10)
        topn_spin.setVisible(False)
        gene_row.addWidget(QLabel("N:"))
        gene_row.addWidget(topn_spin)
        vbox.addLayout(gene_row)

        def _on_gene_mode_changed(idx: int):
            mode = gene_combo.currentText()
            genes_edit.setVisible(mode == "Enter genes…")
            topn_spin.setVisible(mode == "Top-N highly expressed")
        gene_combo.currentIndexChanged.connect(_on_gene_mode_changed)

        # Summary option
        summary_chk = QCheckBox("Include summary text at bottom", dlg)
        summary_chk.setChecked(False)
        vbox.addWidget(summary_chk)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        run_btn = QPushButton("Plot", dlg)
        close_btn = QPushButton("Close", dlg)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(close_btn)
        vbox.addLayout(btn_row)

        # Action handlers
        def _compute_visible_mask() -> np.ndarray:
            if hasattr(self, "_current_mask") and isinstance(getattr(self, "_current_mask"), np.ndarray):
                return self._current_mask
            mask = np.ones(self._adata.n_obs, dtype=bool)
            if getattr(self, "_current_clusters", []):
                if "clusters" in self._adata.obs:
                    mask &= self._adata.obs["clusters"].astype(str).isin(self._current_clusters).values
            if getattr(self, "_current_sources", []):
                if "source" in self._adata.obs:
                    mask &= self._adata.obs["source"].astype(str).isin(self._current_sources).values
            if self._mode == "cell" and getattr(self, "_current_genes", []):
                try:
                    X_sub = self._adata[:, self._current_genes].X
                    if hasattr(X_sub, "toarray"):
                        X_sub = X_sub.toarray()
                    mask &= (X_sub > float(self._expr_threshold.value())).all(axis=1)
                except Exception:
                    pass
            return mask

        def _resolve_gene_ids(raw_list: list[str]) -> list[str]:
            # Case-insensitive mapping using var_names; prefer gene IDs
            varnames = list(map(str, list(self._adata.var_names)))
            lower_map = {str(v).lower(): str(v) for v in varnames}
            # If there is a 'gene_id' column, include it in matching
            if hasattr(self._adata, 'var') and self._adata.var is not None:
                for col in ("gene_id", "gene_ids", "geneID", "geneIds"):
                    if col in self._adata.var.columns:
                        for orig, mapped in zip(self._adata.var_names, self._adata.var[col].astype(str)):
                            lower_map[str(mapped).lower()] = str(orig)
            resolved = []
            missing = []
            for g in raw_list:
                key = str(g).strip().lower()
                if not key:
                    continue
                if key in lower_map:
                    resolved.append(lower_map[key])
                else:
                    missing.append(g)
            if missing:
                QMessageBox.information(dlg, "Missing Genes", f"Not found: {', '.join(map(str, missing))}")
            # De-duplicate while preserving order
            seen = set()
            out = []
            for g in resolved:
                if g not in seen:
                    out.append(g); seen.add(g)
            return out

        def _collect_genes(mode: str, top_n: int) -> list[str]:
            if mode == "Selected in list":
                sel = [itm.text() for itm in self._gene_list.selectedItems()]
                return _resolve_gene_ids(sel)
            if mode == "Enter genes…":
                text = genes_edit.text().strip()
                raw = [t for chunk in text.replace("\n", ",").split(",") for t in chunk.split()] if text else []
                return _resolve_gene_ids(raw)
            # Top-N
            vis_mask = _compute_visible_mask() if scope_combo.currentText() == "Visible cells only" else None
            try:
                if vis_mask is None:
                    # Use selected groups depending on group-by
                    gb = gb_combo.currentText()
                    if gb == "clusters" and getattr(self, "_current_clusters", []):
                        sel_groups = self._current_clusters
                        mask = self._adata.obs["clusters"].astype(str).isin(sel_groups).values
                    elif gb == "source" and getattr(self, "_current_sources", []):
                        sel_groups = self._current_sources
                        mask = self._adata.obs["source"].astype(str).isin(sel_groups).values
                    else:
                        mask = np.ones(self._adata.n_obs, dtype=bool)
                else:
                    mask = vis_mask
                X = self._adata.X
                if hasattr(X, "toarray"):
                    X = X[mask].toarray()
                else:
                    X = X[mask]
                means = np.asarray(X.mean(axis=0)).ravel()
                order = np.argsort(means)[::-1][:top_n]
                return [str(self._adata.var_names[i]) for i in order]
            except Exception as exc:
                QMessageBox.critical(dlg, "Error", f"Failed to compute top genes:\n{exc}")
                return []

        def _build_mask_for_groups(group_by: str, scope: str) -> tuple[np.ndarray, np.ndarray]:
            groups = self._adata.obs[group_by].astype(str)
            if scope == "Visible cells only":
                mask = _compute_visible_mask()
                used_groups = np.sort(groups[mask].unique())
                # Use all cells from those groups (to compare full distributions)
                mask = groups.isin(used_groups).values
                return mask, used_groups
            # All cells from selected groups
            if group_by == "clusters" and getattr(self, "_current_clusters", []):
                used_groups = np.array(sorted(set(map(str, self._current_clusters))))
            elif group_by == "source" and getattr(self, "_current_sources", []):
                used_groups = np.array(sorted(set(map(str, self._current_sources))))
            else:
                used_groups = np.sort(groups.unique())
            mask = groups.isin(used_groups).values
            return mask, used_groups

        def _run():
            group_by = gb_combo.currentText()
            scope = scope_combo.currentText()
            gene_mode = gene_combo.currentText()
            top_n = int(topn_spin.value())

            # Validate required grouping column
            if group_by not in self._adata.obs:
                QMessageBox.information(dlg, "Missing Column", f"AnnData.obs lacks '{group_by}'.")
                return

            genes = _collect_genes(gene_mode, top_n)
            if not genes:
                QMessageBox.information(dlg, "No Genes", "No valid genes to plot.")
                return

            mask, used_groups = _build_mask_for_groups(group_by, scope)
            if mask is None or not mask.any() or len(used_groups) == 0:
                QMessageBox.information(dlg, "No Cells", "No cells match the chosen scope/groups.")
                return

            # Assemble long-form dataframe
            try:
                groups = self._adata.obs[group_by].astype(str)
                X = self._adata[:, genes].X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                frames = []
                for j, g in enumerate(genes):
                    df = pd.DataFrame({
                        "group": groups.values,
                        "expression": X[:, j],
                        "gene": g
                    })
                    frames.append(df[mask])
                df_all = pd.concat(frames, ignore_index=True)
            except Exception as exc:
                QMessageBox.critical(dlg, "Error", f"Failed to prepare data:\n{exc}")
                return

            # Draw in a new dialog with canvas
            plot_dlg = QDialog(dlg)
            plot_dlg.setWindowTitle(f"Violin: {group_by}")
            pvbox = QVBoxLayout(plot_dlg)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            pvbox.addWidget(canvas)

            axes = []
            try:
                import seaborn as sns
                if len(genes) == 1:
                    ax = fig.add_subplot(111)
                    sns.violinplot(data=df_all, x="group", y="expression", ax=ax, cut=0, inner="quartile", scale="width")
                    ax.set_title(f"{genes[0]} by {group_by} ({int(mask.sum()):,} cells)")
                    axes.append(ax)
                else:
                    ncols = min(3, len(genes))
                    nrows = int(np.ceil(len(genes) / ncols))
                    grid = fig.subplots(nrows, ncols, squeeze=False)
                    for i, gene in enumerate(genes):
                        r, c = divmod(i, ncols)
                        ax = grid[r][c]
                        sns.violinplot(data=df_all[df_all["gene"] == gene], x="group", y="expression", ax=ax, cut=0, inner="quartile", scale="width")
                        ax.set_title(gene)
                        axes.append(ax)
                    for k in range(len(genes), nrows * ncols):
                        r, c = divmod(k, ncols)
                        grid[r][c].axis('off')
            except Exception:
                ax = fig.add_subplot(111)
                ordered = sorted(used_groups, key=lambda x: (False, float(x)) if str(x).replace('.', '', 1).isdigit() else (True, str(x)))
                for i, gene in enumerate(genes):
                    ydata = [df_all[(df_all["group"] == grp) & (df_all["gene"] == gene)]["expression"].values for grp in ordered]
                    ax.violinplot(ydata, positions=np.arange(len(ordered)) + i*0.1, widths=0.8/len(genes), showmeans=False, showextrema=False, showmedians=True)
                ax.set_xticks(np.arange(len(ordered)))
                ax.set_xticklabels(ordered, rotation=45, ha='right')
                ax.set_title(f"Expression by {group_by} ({', '.join(genes)})")
                axes.append(ax)

            for ax in axes:
                ax.set_xlabel(group_by.capitalize())
                ax.set_ylabel("Expression")
                ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()

            # Save and close row
            prow = QHBoxLayout()
            if summary_chk.isChecked():
                # Add a simple summary note at bottom
                try:
                    fig.subplots_adjust(bottom=0.22)
                    fig.text(0.01, 0.02, "\n".join([
                        f"Groups: {', '.join(map(str, used_groups))}",
                        f"Genes: {', '.join(genes)}",
                        f"Cells used: {int(mask.sum()):,}",
                    ]), ha='left', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
                    canvas.draw_idle()
                except Exception:
                    pass

            save_btn = QPushButton("Save", plot_dlg)
            def _save_plot():
                fname, _ = QFileDialog.getSaveFileName(plot_dlg, "Save Violin Plot", "violin_plot.png", "PNG files (*.png);")
                if fname:
                    try:
                        fig.savefig(fname, dpi=300, bbox_inches='tight')
                        QMessageBox.information(plot_dlg, "Saved", f"Saved to:\n{fname}")
                    except Exception as exc:
                        QMessageBox.critical(plot_dlg, "Save Error", f"Failed to save plot:\n{exc}")
            save_btn.clicked.connect(_save_plot)
            prow.addWidget(save_btn)
            close2 = QPushButton("Close", plot_dlg)
            close2.clicked.connect(plot_dlg.accept)
            prow.addWidget(close2)
            pvbox.addLayout(prow)

            plot_dlg.resize(1000, 600)
            plot_dlg.exec_()

        run_btn.clicked.connect(_run)
        close_btn.clicked.connect(dlg.accept)
        dlg.resize(520, 180)
        dlg.exec_()

    # ---------------------------------------------------------------------
    # Differential Expression (Volcano Plot)
    # ---------------------------------------------------------------------
    def _show_de_volcano_dialog(self) -> None:
        if self._adata is None:
            QMessageBox.information(self, "No Data", "Load an AnnData file first.")
            return
        # Simple dialog to pick comparison mode and groups
        dlg = QDialog(self)
        dlg.setWindowTitle("DE Analysis (Volcano Plot)")
        layout = QVBoxLayout(dlg)

        form = QFormLayout()
        mode_combo = QComboBox(dlg)
        mode_combo.addItems(["Between Sources/Sections", "Between Clusters (within selected sources)"])
        form.addRow("Comparison:", mode_combo)

        group_a_combo = QComboBox(dlg)
        group_b_combo = QComboBox(dlg)
        # Optional cluster filter when comparing sources
        cluster_combo = QComboBox(dlg)
        if "clusters" in getattr(self._adata, 'obs', {}):
            clust_all = sorted(map(str, self._adata.obs["clusters"].astype(str).unique()),
                               key=lambda x: (0, float(x)) if str(x).replace('.','',1).isdigit() else (1, str(x)))
            for v in clust_all:
                cluster_combo.addItem(v)
        form.addRow("Cluster (sources mode):", cluster_combo)
        # Threshold controls
        # Multiple testing and thresholds
        p_thr_spin = QDoubleSpinBox(dlg)
        p_thr_spin.setDecimals(6)
        p_thr_spin.setRange(1e-12, 1.0)
        p_thr_spin.setSingleStep(0.0001)
        p_thr_spin.setValue(0.05)
        form.addRow("FDR (q) threshold:", p_thr_spin)

        log2fc_thr_spin = QDoubleSpinBox(dlg)
        log2fc_thr_spin.setDecimals(2)
        log2fc_thr_spin.setRange(0.0, 50.0)
        log2fc_thr_spin.setSingleStep(0.1)
        log2fc_thr_spin.setValue(1.0)
        form.addRow("|log2FC| threshold:", log2fc_thr_spin)

        # Minimum percent expressed filter
        min_pct_spin = QDoubleSpinBox(dlg)
        min_pct_spin.setDecimals(1)
        min_pct_spin.setRange(0.0, 100.0)
        min_pct_spin.setSingleStep(1.0)
        min_pct_spin.setValue(0.0)
        form.addRow("Min % expressed in either group:", min_pct_spin)

        # Expression source (X or a layer)
        layer_combo = QComboBox(dlg)
        layer_combo.addItem("X (current)")
        try:
            if hasattr(self._adata, 'layers') and len(self._adata.layers.keys()) > 0:
                for k in self._adata.layers.keys():
                    layer_combo.addItem(str(k))
        except Exception:
            pass
        form.addRow("Expression matrix:", layer_combo)

        label_top_spin = QSpinBox(dlg)
        label_top_spin.setRange(0, 200)
        label_top_spin.setValue(20)
        form.addRow("Max genes to label:", label_top_spin)

        def populate_groups():
            group_a_combo.clear(); group_b_combo.clear()
            if mode_combo.currentIndex() == 0:
                # Source comparison
                if "source" not in self._adata.obs:
                    group_a_combo.addItem("<missing 'source'>"); group_b_combo.addItem("<missing 'source'>")
                else:
                    vals = sorted(map(str, self._adata.obs["source"].astype(str).unique()))
                    for v in vals:
                        group_a_combo.addItem(v); group_b_combo.addItem(v)
            else:
                # Cluster comparison; limit to currently selected/visible
                if "clusters" not in self._adata.obs:
                    group_a_combo.addItem("<missing 'clusters'>"); group_b_combo.addItem("<missing 'clusters'>")
                else:
                    clust_vals = self._adata.obs["clusters"].astype(str)
                    if self._current_sources and "source" in self._adata.obs:
                        mask = self._adata.obs["source"].astype(str).isin(self._current_sources).values
                        clust_vals = clust_vals[mask]
                    uniq = sorted(clust_vals.unique(), key=lambda x: (0, float(x)) if str(x).replace('.','',1).isdigit() else (1, str(x)))
                    for v in uniq:
                        group_a_combo.addItem(v); group_b_combo.addItem(v)

        populate_groups()
        mode_combo.currentIndexChanged.connect(populate_groups)
        form.addRow("Group A:", group_a_combo)
        form.addRow("Group B:", group_b_combo)

        # Sources scope for cluster comparison
        use_sel_sources_chk = QCheckBox("Use only chosen sources (else all)")
        form.addRow("Sources scope:", use_sel_sources_chk)
        chosen_sources: list[str] = []
        def pick_sources():
            try:
                sources = sorted(map(str, self._adata.obs["source"].astype(str).unique())) if "source" in self._adata.obs else []
            except Exception:
                sources = []
            if not sources:
                QMessageBox.information(dlg, "No Sources", "No 'source' column found.")
                return
            # Reuse SectionSelectDialog for picking
            try:
                sel_dlg = SectionSelectDialog(sources, self)
            except Exception:
                sel_dlg = None
            if sel_dlg and sel_dlg.exec_() == QDialog.Accepted:
                sel = sel_dlg.get_selected_sources()
                if sel:
                    chosen_sources.clear(); chosen_sources.extend(sel)
                    use_sel_sources_chk.setChecked(True)
        pick_btn = QPushButton("Pick sources…", dlg)
        pick_btn.clicked.connect(pick_sources)
        form.addRow(" ", pick_btn)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run", dlg)
        close_btn = QPushButton("Close", dlg)
        btn_row.addStretch(); btn_row.addWidget(run_btn); btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def run_volcano():
            a = group_a_combo.currentText(); b = group_b_combo.currentText()
            if a == b:
                QMessageBox.information(dlg, "Invalid", "Group A and B must be different.")
                return
            try:
                mode_idx = mode_combo.currentIndex()
                cluster_filter = cluster_combo.currentText() if mode_idx == 0 and cluster_combo.count() > 0 else None
                sources_subset = None
                if mode_idx == 1 and use_sel_sources_chk.isChecked():
                    if len(chosen_sources) > 0:
                        sources_subset = list(chosen_sources)
                    elif len(self._current_sources) > 0:
                        sources_subset = list(self._current_sources)
                self._compute_and_show_volcano(
                    mode_idx, a, b,
                    p_thr_spin.value(), log2fc_thr_spin.value(), label_top_spin.value(),
                    cluster_filter, sources_subset,
                    min_pct_spin.value(), layer_combo.currentText()
                )
            except Exception as exc:
                QMessageBox.critical(dlg, "DE Error", f"Failed to compute DE:\n{exc}")

        run_btn.clicked.connect(run_volcano)
        close_btn.clicked.connect(dlg.accept)
        dlg.resize(500, 200)
        dlg.exec_()

    def _compute_and_show_volcano(self, mode_index: int, group_a: str, group_b: str,
                                  q_threshold: float, log2fc_threshold: float, max_labels: int,
                                  cluster_filter: Optional[str] = None, sources_subset: Optional[list] = None,
                                  min_pct_expressed: float = 0.0, which_layer: str = "X (current)") -> None:
        # Build mask and labels
        if mode_index == 0:
            # Sources/sections comparison
            if "source" not in self._adata.obs:
                QMessageBox.information(self, "Missing", "Dataset has no 'source' in obs.")
                return
            mask = self._adata.obs["source"].astype(str).isin([group_a, group_b]).values
            # Optional cluster filter when comparing sources
            if cluster_filter is not None and "clusters" in self._adata.obs:
                mask &= self._adata.obs["clusters"].astype(str).isin([cluster_filter]).values
            labels = self._adata.obs["source"].astype(str).values
        else:
            # Cluster comparison, optionally within selected sources
            if "clusters" not in self._adata.obs:
                QMessageBox.information(self, "Missing", "Dataset has no 'clusters' in obs.")
                return
            mask = self._adata.obs["clusters"].astype(str).isin([group_a, group_b]).values
            # Restrict to chosen sources if provided
            if sources_subset and "source" in self._adata.obs:
                mask &= self._adata.obs["source"].astype(str).isin(list(sources_subset)).values
            labels = self._adata.obs["clusters"].astype(str).values

        if not mask.any():
            QMessageBox.information(self, "No Cells", "No cells for the selected groups.")
            return

        # Use chosen matrix (X or a named layer)
        adata_sel = self._adata[mask]
        if which_layer != "X (current)":
            try:
                X = adata_sel.layers[which_layer]
            except Exception:
                QMessageBox.information(self, "Layer Missing", f"Layer '{which_layer}' not found. Using X.")
                X = adata_sel.X
        else:
            X = adata_sel.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        labels = labels[mask]

        import numpy as np
        import pandas as pd
        # Split into two groups
        idx_a = np.where(labels == group_a)[0]
        idx_b = np.where(labels == group_b)[0]
        if idx_a.size == 0 or idx_b.size == 0:
            QMessageBox.information(self, "Insufficient", "One of the groups has zero cells.")
            return

        # Compute log2 fold change and p-values (nonparametric preferred)
        from scipy import stats
        try:
            mean_a = X[idx_a].mean(axis=0)
            mean_b = X[idx_b].mean(axis=0)
            # Add small epsilon to avoid log of zero
            eps = 1e-8
            log2fc = np.log2((mean_a + eps) / (mean_b + eps))
            # Mann–Whitney U; fallback to t-test if needed
            try:
                pvals = np.array([stats.mannwhitneyu(X[idx_a, j], X[idx_b, j], alternative='two-sided').pvalue for j in range(X.shape[1])])
            except Exception:
                _, pvals = stats.ttest_ind(X[idx_a], X[idx_b], axis=0, equal_var=False)
            pvals = np.nan_to_num(pvals, nan=1.0)
        except Exception as exc:
            QMessageBox.critical(self, "Stats Error", f"Failed to compute statistics:\n{exc}")
            return

        # Assemble DataFrame
        df = pd.DataFrame({
            'gene': self._adata.var_names,
            'log2FC': np.asarray(log2fc).ravel(),
            'pval': np.asarray(pvals).ravel(),
        })
        # Filter genes by min percent expressed in either group
        try:
            if min_pct_expressed > 0.0:
                pct_a = (X[idx_a] > 0).mean(axis=0) * 100.0
                pct_b = (X[idx_b] > 0).mean(axis=0) * 100.0
                keep = (pct_a >= min_pct_expressed) | (pct_b >= min_pct_expressed)
                keep = np.asarray(keep).ravel()
                df = df.loc[keep].reset_index(drop=True)
        except Exception:
            pass

        # Benjamini–Hochberg FDR
        try:
            from statsmodels.stats.multitest import multipletests
            _, qvals, _, _ = multipletests(df['pval'].values, method='fdr_bh')
            df['qval'] = qvals
        except Exception:
            df['qval'] = df['pval']
        df['neglog10q'] = -np.log10(df['qval'] + 1e-300)

        # Volcano plot dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Volcano: {group_a} vs {group_b}")
        vbox = QVBoxLayout(dlg)
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        vbox.addWidget(canvas)
        ax = fig.add_subplot(111)
        # Determine significance based on thresholds (use q-values)
        sig_a = (df['qval'] <= q_threshold) & (df['log2FC'] >= log2fc_threshold)
        sig_b = (df['qval'] <= q_threshold) & (df['log2FC'] <= -log2fc_threshold)
        rest = ~(sig_a | sig_b)

        ax.scatter(df.loc[rest, 'log2FC'], df.loc[rest, 'neglog10q'], s=6, c='gray', alpha=0.5)
        ax.scatter(df.loc[sig_a, 'log2FC'], df.loc[sig_a, 'neglog10q'], s=10, c='red', alpha=0.9, label=f"Up in {group_a}")
        ax.scatter(df.loc[sig_b, 'log2FC'], df.loc[sig_b, 'neglog10q'], s=10, c='blue', alpha=0.9, label=f"Up in {group_b}")
        # Draw threshold lines
        ax.axvline(log2fc_threshold, color='red', linestyle='--', linewidth=1)
        ax.axvline(-log2fc_threshold, color='blue', linestyle='--', linewidth=1)
        ax.axhline(-np.log10(q_threshold), color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('log2 Fold Change')
        ax.set_ylabel('-log10 q-value')
        ax.set_title(f"{group_a} vs {group_b}")
        # Adjust axes to prioritize significant region visibility
        try:
            x_min = min(df.loc[sig_b, 'log2FC'].min() if sig_b.any() else -log2fc_threshold*1.2, df['log2FC'].min())
            x_max = max(df.loc[sig_a, 'log2FC'].max() if sig_a.any() else log2fc_threshold*1.2, df['log2FC'].max())
            y_min = 0
            y_max = max(df.loc[sig_a | sig_b, 'neglog10q'].max() if (sig_a.any() or sig_b.any()) else -np.log10(q_threshold)*1.5, df['neglog10q'].quantile(0.98))
            # Squeeze non-significant sides a bit
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max * 1.05)
        except Exception:
            pass

        # Label top genes (by neglog10q) within each significant side
        try:
            def label_side(mask, color):
                sub = df.loc[mask].sort_values('neglog10q', ascending=False).head(max_labels)
                for _, row in sub.iterrows():
                    ax.text(row['log2FC'], row['neglog10q'], str(row['gene']), color=color, fontsize=8,
                            ha='left', va='bottom')
            if max_labels > 0:
                label_side(sig_a, 'red')
                label_side(sig_b, 'blue')
        except Exception:
            pass

        ax.legend(loc='upper right', fontsize=8, frameon=False)
        fig.tight_layout()

        # Save button
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save", dlg)
        summary_btn = QPushButton("Summary", dlg)
        close_btn = QPushButton("Close", dlg)
        btn_row.addStretch(); btn_row.addWidget(save_btn); btn_row.addWidget(summary_btn); btn_row.addWidget(close_btn)
        vbox.addLayout(btn_row)

        def _save_volcano():
            fname, _ = QFileDialog.getSaveFileName(dlg, "Save Volcano Plot", "volcano.png", "PNG files (*.png)")
            if fname:
                try:
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    QMessageBox.information(dlg, "Saved", f"Saved to:\n{fname}")
                except Exception as exc:
                    QMessageBox.critical(dlg, "Save Error", f"Failed to save volcano plot:\n{exc}")
        save_btn.clicked.connect(_save_volcano)

        def _show_summary_dialog():
            total = int(len(df))
            n_a = int(sig_a.sum())
            n_b = int(sig_b.sum())
            n_rest = int(rest.sum())
            dlg2 = QDialog(self)
            dlg2.setWindowTitle("DE Summary")
            lay = QVBoxLayout(dlg2)
            lbl = QLabel(
                f"Comparison: {group_a} vs {group_b}\n"
                f"Thresholds: |log2FC| ≥ {log2fc_threshold:g}, q ≤ {q_threshold:g}\n\n"
                f"Up in {group_a}: {n_a} ({(n_a/total*100 if total>0 else 0):.1f}%)\n"
                f"Up in {group_b}: {n_b} ({(n_b/total*100 if total>0 else 0):.1f}%)\n"
                f"Not significant: {n_rest} ({(n_rest/total*100 if total>0 else 0):.1f}%)\n"
                f"Genes tested: {total}"
            )
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

            # Tabs with sortable tables for each category
            tabs = QTabWidget(dlg2)
            lay.addWidget(tabs, 1)

            def make_table(dataframe):
                tbl = QTableWidget()
                tbl.setColumnCount(4)
                tbl.setHorizontalHeaderLabels(["gene", "log2FC", "qval", "-log10(q)"])
                tbl.setRowCount(len(dataframe))
                for i, (_, r) in enumerate(dataframe.iterrows()):
                    tbl.setItem(i, 0, QTableWidgetItem(str(r.get('gene', ''))))
                    tbl.setItem(i, 1, QTableWidgetItem(f"{float(r.get('log2FC', 0)):.3f}"))
                    qv = float(r.get('qval', r.get('pval', 1.0)))
                    tbl.setItem(i, 2, QTableWidgetItem(f"{qv:.3g}"))
                    tbl.setItem(i, 3, QTableWidgetItem(f"{-np.log10(max(qv,1e-300)):.2f}"))
                tbl.setSortingEnabled(True)
                tbl.resizeColumnsToContents()
                return tbl

            df_a = df.loc[sig_a].copy().sort_values(['qval','log2FC'], ascending=[True, False]) if 'qval' in df.columns else df.loc[sig_a].copy()
            df_b = df.loc[sig_b].copy().sort_values(['qval','log2FC'], ascending=[True, True]) if 'qval' in df.columns else df.loc[sig_b].copy()
            df_ns = df.loc[rest].copy().sort_values(['qval'] if 'qval' in df.columns else ['pval'])

            tabs.addTab(make_table(df_a), f"Up in {group_a} ({n_a})")
            tabs.addTab(make_table(df_b), f"Up in {group_b} ({n_b})")
            tabs.addTab(make_table(df_ns), f"Not significant ({n_rest})")

            # Optional: save summary table
            btns = QHBoxLayout()
            export_btn = QPushButton("Save Table", dlg2)
            close2_btn = QPushButton("Close", dlg2)
            btns.addStretch(); btns.addWidget(export_btn); btns.addWidget(close2_btn)
            lay.addLayout(btns)

            def _export_table():
                fname, _ = QFileDialog.getSaveFileName(dlg2, "Save DE Table", "de_summary.csv", "CSV files (*.csv)")
                if not fname:
                    return
                try:
                    # Build an assignment of category
                    cat = np.full(len(df), 'not_significant', dtype=object)
                    cat[sig_a.values if hasattr(sig_a, 'values') else sig_a] = f'up_{group_a}'
                    cat[sig_b.values if hasattr(sig_b, 'values') else sig_b] = f'up_{group_b}'
                    out = df.copy()
                    out['category'] = cat
                    # Keep essential columns
                    cols = ['gene', 'log2FC', 'qval', 'category'] if 'qval' in out.columns else ['gene','log2FC','pval','category']
                    out.to_csv(fname, index=False, columns=cols)
                    QMessageBox.information(dlg2, "Saved", f"Summary saved to:\n{fname}")
                except Exception as exc:
                    QMessageBox.critical(dlg2, "Save Error", f"Failed to save summary:\n{exc}")

            export_btn.clicked.connect(_export_table)
            close2_btn.clicked.connect(dlg2.accept)
            dlg2.resize(800, 500)
            dlg2.exec_()

        summary_btn.clicked.connect(_show_summary_dialog)
        close_btn.clicked.connect(dlg.accept)
        dlg.resize(900, 700)
        dlg.exec_()

    def _update_selection_summary(self, mask: Optional[np.ndarray]) -> None:
        """Update the side-panel summary of what's currently shown.
        If mask is None (non-cell modes), summarize from current selections only.
        """
        try:
            if self._adata is None:
                self._summary_label.setText("No data loaded.")
                return
            # If mask not provided, build it roughly from current filters for counts
            if mask is None:
                tmp_mask = np.ones(self._adata.n_obs, dtype=bool)
                if self._current_clusters and "clusters" in self._adata.obs:
                    tmp_mask &= self._adata.obs["clusters"].astype(str).isin(self._current_clusters).values
                if self._current_sources and "source" in self._adata.obs:
                    tmp_mask &= self._adata.obs["source"].astype(str).isin(self._current_sources).values
                if self._current_genes:
                    try:
                        X_sub = self._adata[:, self._current_genes].X
                        if hasattr(X_sub, "toarray"):
                            X_sub = X_sub.toarray()
                        tmp_mask &= (X_sub > float(self._expr_threshold.value())).all(axis=1)
                    except Exception:
                        pass
                mask = tmp_mask

            n_show = int(mask.sum()) if mask is not None else 0
            # Visible clusters under mask with counts
            clusters_txt = "NA"
            if "clusters" in self._adata.obs and mask is not None and mask.any():
                clust_vals = self._adata.obs.loc[mask, "clusters"].astype(str)
                counts = clust_vals.value_counts()
                # Sort by cluster label for stable ordering
                try:
                    # Attempt natural numeric sort when labels are numeric strings
                    sorted_labels = sorted(counts.index, key=lambda x: (False, float(x)) if x.replace('.', '', 1).isdigit() else (True, x))
                except Exception:
                    sorted_labels = sorted(counts.index)
                clusters_txt = ", ".join([f"{lbl} ({int(counts[lbl])})" for lbl in sorted_labels]) if len(counts) else "NA"

            # Visible sections
            sections_txt = "NA"
            if "source" in self._adata.obs and mask is not None and mask.any():
                src_vals = self._adata.obs.loc[mask, "source"].astype(str)
                uniqs = np.sort(src_vals.unique())
                sections_txt = ", ".join(map(str, uniqs)) if len(uniqs) else "NA"

            # Gene filter text
            genes_txt = ", ".join(self._current_genes) if self._current_genes else "—"
            thr = self._expr_threshold.value()
            thr_txt = f"expr > {thr:g}" if self._current_genes else ""

            lines = [
                f"Shown cells: {n_show:,}",
                f"Clusters: {clusters_txt}",
                f"Sections: {sections_txt}",
                (f"Genes: {genes_txt}  {thr_txt}".strip()),
            ]
            self._summary_label.setText("\n".join(lines))
        except Exception as _:
            # Fail safe: do not interrupt rendering due to summary
            pass

    # -------------------------------------------------------------------------
    # Drag & drop events
    # -------------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt naming)
        mime: QMimeData = event.mimeData()
        if mime.hasUrls() and any(url.toLocalFile().endswith(".h5ad") for url in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 (Qt naming)
        for url in event.mimeData().urls():
            filepath = pathlib.Path(url.toLocalFile())
            if filepath.suffix == ".h5ad":
                self._load_adata(filepath)
                break
        event.acceptProposedAction()

    # -------------------------------------------------------------------------
    # UI helpers
    # -------------------------------------------------------------------------
    def _open_file_dialog(self) -> None:
        """Open a file dialog for users who prefer clicking."""
        filename, _ = QFileDialog.getOpenFileName(self, "Open .h5ad file", str(pathlib.Path.home()), "AnnData files (*.h5ad)")
        if filename:
            self._load_adata(pathlib.Path(filename))

    def _open_gene_adata_dialog(self):
        """Open file dialog to load gene-level AnnData (.h5ad)."""
        filename, _ = QFileDialog.getOpenFileName(self, "Open gene spots .h5ad", str(pathlib.Path.home()), "AnnData files (*.h5ad)")
        if filename:
            try:
                self._gene_adata = ad.read_h5ad(filename)
                self._populate_genes_from_geneadata()
                QMessageBox.information(self, "Loaded", f"Loaded gene data: {pathlib.Path(filename).name}")
            except Exception as exc:
                QMessageBox.critical(self, "Load Error", f"Failed to load gene AnnData:\n{exc}")

    def _open_color_dialog(self):
        """Open color scheme selection dialog."""
        dialog = ColorSchemeDialog(self, self._color_scheme, self._adata)
        if dialog.exec_() == QDialog.Accepted:
            self._color_scheme = dialog.get_selected_scheme()
            # Re-render with new color scheme
            if self._mode == "cell":
                self._render_spatial()
            elif self._mode == "gene":
                self._render_gene_mode()
            elif self._mode == "gene_spots":
                self._render_gene_only()

    # -------------------------------------------------------------------------
    # AnnData loading & plotting
    # -------------------------------------------------------------------------
    def _load_adata(self, path: pathlib.Path) -> None:
        try:
            self._status_label.setText(f"Loading {path.name} …")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            self._adata = ad.read_h5ad(str(path))
            self._status_label.setText(f"Loaded {path.name} (n = {self._adata.n_obs:,} cells)")
            self._populate_controls()
            self._render_spatial()
        except Exception as exc:  # pragma: no cover
            self._status_label.setText(f"Failed to load file: {exc}")
            import traceback
            traceback.print_exc()

    def _render_spatial(self) -> None:
        """Render the spatial coordinates of the AnnData object."""
        if self._adata is None:
            return

        # Validate spatial coordinates
        coords_key = "spatial"
        if coords_key not in self._adata.obsm:
            self._status_label.setText("The AnnData object lacks obsm['spatial'] coordinates.")
            return

        coords_full = self._adata.obsm[coords_key].astype(float)

        # Handle 2D coordinates by appending a z=0 column
        if coords_full.shape[1] == 2:
            z = np.zeros((coords_full.shape[0], 1), dtype=coords_full.dtype)
            coords_full = np.hstack([coords_full, z])

            # If multiple sources exist, offset each source along z by +10, +20, ...
            if "source" in self._adata.obs:
                sources = self._adata.obs["source"].astype(str)
                unique_sources = pd.Index(sources.unique())
                offset_lookup = {src: i * 100.0 for i, src in enumerate(unique_sources)}
                offsets = sources.map(offset_lookup).to_numpy(dtype=coords_full.dtype)
                coords_full[:, 2] += offsets

        elif coords_full.shape[1] != 3:
            self._status_label.setText("spatial matrix must have 2 (x,y) or 3 (x,y,z) columns.")
            return

        # Save current camera to preserve perspective
        cam_position = None
        if self._has_initial_camera:
            cam_position = self._plotter_widget.camera_position

        # Apply filters ---------------------------------------------------------
        mask = np.ones(self._adata.n_obs, dtype=bool)

        # Cluster filter
        if self._current_clusters:
            if "clusters" in self._adata.obs:
                mask &= self._adata.obs["clusters"].astype(str).isin(self._current_clusters).values

        # Source filter
        if self._current_sources:
            if "source" in self._adata.obs:
                mask &= self._adata.obs["source"].astype(str).isin(self._current_sources).values

        # Gene filter (cells must express ALL selected genes above threshold)
        if self._current_genes:
            try:
                X_sub = self._adata[:, self._current_genes].X
                thresh = self._expr_threshold.value()
                # Handle sparse matrices transparently
                if hasattr(X_sub, "toarray"):
                    X_sub = X_sub.toarray()
                mask &= (X_sub > thresh).all(axis=1)
            except KeyError:
                pass  # gene names not present

        if not mask.any():
            self._status_label.setText("No cells match the current filters.")
            self._plotter_widget.clear()
            return

        self._coords_all = coords_full  # store for picking

        # Build KD-tree for fast spatial lookups (analysis mode)
        try:
            self._kd_tree = cKDTree(coords_full)
        except Exception:
            self._kd_tree = None

        coords_sel = coords_full[mask]
        coords_other = coords_full[~mask]

        self._plotter_widget.clear()

        # Plot non-selected cells in gray
        if coords_other.shape[0] > 0:
            self._plotter_widget.add_mesh(
                pv.PolyData(coords_other),
                color=(150, 150, 150),
                point_size=3,
                render_points_as_spheres=True,
                opacity=0.1,  # 90% transparency for grayed cells (opacity 0.1)
            )

        # Colour points ---------------------------------------------------------
        if "clusters" in self._adata.obs and coords_sel.shape[0] > 0:
            clusters_series = self._adata.obs["clusters"].astype(str)
            clusters_sel = clusters_series[mask]

            colours = None  # will populate below

            # Generate colors based on selected color scheme
            colours = None
            all_cats = np.sort(np.unique(clusters_series))
            
            if self._color_scheme == "anndata" and "clusters_colors" in self._adata.obs:
                try:
                    # Get colors for selected cells directly from obs
                    colors_selected = self._adata.obs["clusters_colors"][mask]
                    # Convert color strings to RGB values
                    colours = np.vstack([mcolors.to_rgb(str(c)) for c in colors_selected]) * 255
                except Exception:
                    colours = None  # fallback if color parsing fails
            
            # Apply selected color scheme or fallback
            if colours is None:
                if self._color_scheme == "plotly_d3":
                    palette = generate_plotly_extended_palette(len(all_cats))
                elif self._color_scheme == "custom_turbo":
                    palette = generate_custom_turbo_palette(len(all_cats))
                elif self._color_scheme == "sns_palette":
                    palette = generate_sns_palette(len(all_cats))
                elif self._color_scheme == "milliomics":
                    palette = generate_milliomics_palette(len(all_cats))
                else:  # fallback to plotly_d3
                    palette = generate_plotly_extended_palette(len(all_cats))
                
                lookup = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}
                colours = np.vstack([lookup[c] for c in clusters_sel]) * 255
        else:
            colours = np.full_like(coords_sel, fill_value=255)

        cloud_sel = pv.PolyData(coords_sel)
        cloud_sel["colors"] = colours
        self._plotter_widget.add_mesh(
            cloud_sel,
            scalars="colors",
            rgb=True,
            point_size=5,
            render_points_as_spheres=True,
            opacity=self._point_opacity,
        )

        # Invisible cloud for precise point picking ---------------------
        if hasattr(self, "_full_pick_actor") and self._full_pick_actor is not None:
            try:
                self._plotter_widget.remove_actor(self._full_pick_actor)
            except Exception:
                pass

        self._full_pick_actor = self._plotter_widget.add_mesh(
            pv.PolyData(coords_full),
            color=(1, 1, 1),  # color irrelevant
            opacity=0.0,  # invisible
            point_size=10,
            pickable=True,
            name="_pick_cloud",
            render_points_as_spheres=False,
        )

        self._add_bounding_box(coords_full)
        self._ensure_orientation_widget()
        self._plotter_widget.set_background("black")

        # Restore camera or reset if first time
        if cam_position is not None:
            self._plotter_widget.camera_position = cam_position
        else:
            # Reset camera then set to a top-down view so initial display is not rotated
            self._plotter_widget.reset_camera()

            # Compute top-down camera parameters based on overall bounds
            try:
                bounds = coords_full.min(axis=0), coords_full.max(axis=0)
                (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                z_center = (zmin + zmax) / 2.0
                view_dist = max(xmax - xmin, ymax - ymin) * 1.5 + (zmax - zmin)

                camera_pos = [x_center, y_center, z_max := zmax + view_dist]
                focal_point = [x_center, y_center, z_center]
                view_up = [0, 1, 0]
                self._plotter_widget.camera_position = [camera_pos, focal_point, view_up]
                self._plotter_widget.camera.parallel_projection = True
            except Exception:
                # Fallback: leave default orientation if anything fails
                pass

            self._has_initial_camera = True

        self._plotter_widget.render()

        # Save mask for hover use
        self._current_mask = mask

        # Update selection summary text
        self._update_selection_summary(mask)

    # -------------------------------------------------------------------------
    # Gene CSV loading
    # -------------------------------------------------------------------------
    def _populate_genes_from_geneadata(self):
        self._gene_list.clear()
        for g in sorted(self._gene_adata.var_names):
            self._gene_list.addItem(g)

    # ---------------------------------------------------------------------
    # Gene mode rendering
    # ---------------------------------------------------------------------
    def _render_gene_mode(self):
        selected_genes = [itm.text() for itm in self._gene_list.selectedItems()]
        if not selected_genes:
            self._plotter_widget.clear(); self._plotter_widget.render(); return

        self._plotter_widget.clear()
        for gene in selected_genes:
            data = self._gene_adata[gene]
            if data["neurite"].size:
                self._plotter_widget.add_mesh(
                    pv.PolyData(data["neurite"]),
                    color="lime",
                    point_size=6,
                    render_points_as_spheres=True,
                    opacity=self._point_opacity,
                )
            if data["soma"].size:
                self._plotter_widget.add_mesh(
                    pv.PolyData(data["soma"]),
                    color="red",
                    point_size=6,
                    render_points_as_spheres=True,
                    opacity=self._point_opacity,
                )

        self._plotter_widget.set_background("black")
        self._plotter_widget.reset_camera()
        self._plotter_widget.render()

    # ---------------------------------------------------------------------
    # Gene-only spot renderer
    # ---------------------------------------------------------------------
    def _render_gene_only(self):
        selected_genes = [itm.text() for itm in self._gene_list.selectedItems()]
        if not selected_genes:
            self._plotter_widget.clear(); self._plotter_widget.render(); return

        if self._gene_adata is None:
            # prompt load
            self._open_gene_adata_dialog()
            if self._gene_adata is None:
                return

        self._plotter_widget.clear()

        col = 'hb_gene_name' if 'hb_gene_name' in self._gene_adata.obs.columns else (
            'gene_name' if 'gene_name' in self._gene_adata.obs.columns else None)
        if col is None:
            QMessageBox.information(self, "Missing Column", "Gene AnnData must have 'hb_gene_name' or 'gene_name' in obs.")
            return

        for gene in selected_genes:
            mask = self._gene_adata.obs[col] == gene
            if not mask.any():
                continue
            coords = self._gene_adata.obsm['spatial'][mask.values]
            self._plotter_widget.add_mesh(
                pv.PolyData(coords),
                color="white",
                point_size=6,
                render_points_as_spheres=True,
                opacity=self._point_opacity,
            )

        self._plotter_widget.set_background("black")
        self._plotter_widget.reset_camera()
        self._plotter_widget.render()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _add_bounding_box(self, coords: np.ndarray) -> None:
        """Draw a white bounding box around the dataset."""
        bounds = [coords[:, 0].min(), coords[:, 0].max(),
                  coords[:, 1].min(), coords[:, 1].max(),
                  coords[:, 2].min(), coords[:, 2].max()]
        box = pv.Box(bounds=bounds)
        self._plotter_widget.add_mesh(box, color="white", style="wireframe", opacity=0.5)

    # ---------------------------------------------------------------------
    # Controls helpers
    # ---------------------------------------------------------------------
    def _populate_controls(self) -> None:
        """Populate cluster and gene lists after a file is loaded."""
        # Clear lists first
        self._cluster_list.clear()
        self._gene_list.clear()

        if self._adata is None:
            return

        # Populate clusters (display annotation in brackets if present)
        if "clusters" in self._adata.obs:
            for cl in np.sort(self._adata.obs["clusters"].astype(str).unique()):
                text = self._format_cluster_item_text(str(cl))
                item = QListWidgetItem(text)
                # Store raw cluster id for logic independent of decorated text
                try:
                    item.setData(Qt.UserRole, str(cl))
                except Exception:
                    pass
                self._cluster_list.addItem(item)

        # Populate sections (sources)
        if "source" in self._adata.obs:
            for src in np.sort(self._adata.obs["source"].astype(str).unique()):
                item = QListWidgetItem(str(src))
                self._section_list.addItem(item)

        # Populate genes (might be large – limit to 5000 for UI responsiveness)
        max_genes = 20000
        genes_iter = (str(g) for g in self._adata.var_names[:max_genes])
        for g in genes_iter:
            item = QListWidgetItem(g)
            self._gene_list.addItem(item)

    def _update_filters(self) -> None:
        """Update selection state and re-render plot."""
        self._current_clusters = self._get_selected_clusters()
        self._current_sources = [itm.text() for itm in self._section_list.selectedItems()]
        self._current_genes = [itm.text() for itm in self._gene_list.selectedItems()]
        if self._mode == "cell":
            self._render_spatial()
        elif self._mode == "gene":
            self._render_gene_mode()
        elif self._mode == "gene_spots":
            self._render_gene_only()

    # ---------------------------------------------------------------------
    # Search helpers
    # ---------------------------------------------------------------------
    def _filter_list_widget(self, list_widget: QListWidget, term: str) -> None:
        term = term.lower()
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item.setHidden(term not in item.text().lower())

    def _perform_cluster_search(self) -> None:
        self._filter_list_widget(self._cluster_list, self._cluster_search_line.text())

    def _format_cluster_item_text(self, cluster_label: str) -> str:
        """Return display text for a cluster item including annotation in brackets if exists."""
        try:
            if hasattr(self, "_cluster_annotations") and cluster_label in self._cluster_annotations:
                anno = self._cluster_annotations.get(cluster_label, "").strip()
                if anno:
                    return f"{cluster_label} [{anno}]"
        except Exception:
            pass
        return str(cluster_label)

    def _get_selected_clusters(self) -> list[str]:
        """Read selected clusters using stored raw IDs when available."""
        clusters: list[str] = []
        for itm in self._cluster_list.selectedItems():
            raw = itm.data(Qt.UserRole)
            clusters.append(raw if raw is not None else itm.text().split("[")[0].strip())
        return clusters

    def _perform_gene_search(self) -> None:
        self._filter_list_widget(self._gene_list, self._gene_search_line.text())

    def _perform_section_search(self) -> None:
        self._filter_list_widget(self._section_list, self._section_search_line.text())

    # ---------------------------------------------------------------------
    # Orientation axes
    # ---------------------------------------------------------------------
    def _refresh_cluster_list_labels(self) -> None:
        """Re-apply bracketed annotations to cluster list items without changing selection."""
        # Remember current selections by raw id if stored
        selected_ids = set()
        for itm in self._cluster_list.selectedItems():
            raw = itm.data(Qt.UserRole)
            selected_ids.add(raw if raw is not None else itm.text().split("[")[0].strip())

        # Update each item's text based on current annotations
        for i in range(self._cluster_list.count()):
            itm = self._cluster_list.item(i)
            raw = itm.data(Qt.UserRole)
            raw_str = raw if raw is not None else itm.text().split("[")[0].strip()
            itm.setText(self._format_cluster_item_text(raw_str))

        # Restore selection by raw id
        self._cluster_list.clearSelection()
        for i in range(self._cluster_list.count()):
            itm = self._cluster_list.item(i)
            raw = itm.data(Qt.UserRole)
            raw_str = raw if raw is not None else itm.text().split("[")[0].strip()
            if raw_str in selected_ids:
                itm.setSelected(True)

    def _ensure_orientation_widget(self) -> None:
        if not self._orientation_added:
            try:
                # Use white-colored axes due to black background
                self._plotter_widget.add_axes(
                    interactive=False,
                    line_width=2,
                    color='white'
                )
                self._orientation_added = True
            except Exception:
                pass  # fallback silently if function unavailable

    # ---------------------------------------------------------------------
    # Section navigation helpers
    # ---------------------------------------------------------------------
    def _section_up(self) -> None:
        if self._section_list.count() == 0:
            return
        current_rows = [index.row() for index in self._section_list.selectedIndexes()]
        row = current_rows[0] if current_rows else 0
        new_row = max(row - 1, 0)
        self._section_list.clearSelection()
        self._section_list.setCurrentRow(new_row)
        if self._mode == "cell":
            self._render_spatial()
        else:
            self._render_gene_mode()

    def _section_down(self) -> None:
        if self._section_list.count() == 0:
            return
        current_rows = [index.row() for index in self._section_list.selectedIndexes()]
        row = current_rows[0] if current_rows else -1
        new_row = min(row + 1, self._section_list.count() - 1)
        self._section_list.clearSelection()
        self._section_list.setCurrentRow(new_row)
        if self._mode == "cell":
            self._render_spatial()
        else:
            self._render_gene_mode()

    # ---------------------------------------------------------------------
    # Mouse event handling
    # ---------------------------------------------------------------------
    def _on_mouse_move(self, obj, event):
        x, y = self._plotter_widget.interactor.GetEventPosition()
        if self._picker.Pick(x, y, 0, self._plotter_widget.renderer) == 0:
            self._clear_hover()
            return

        idx = self._picker.GetPointId()
        if idx == -1:
            self._clear_hover()
            return

        if self._hover_idx == idx:
            return  # same cell as before
        pick_pt = np.array(self._coords_all[idx])
        self._hover_idx = idx
        self._display_hover(idx, pick_pt)

    def _clear_hover(self):
        self._hover_idx = -1
        self._display_hover(-1, None)

    def _display_hover(self, idx, pick_pt):
        if idx == -1 or pick_pt is None:
            # clear
            if self._highlight_actor is not None:
                try:
                    self._plotter_widget.remove_actor(self._highlight_actor)
                except Exception:
                    pass
                self._highlight_actor = None
                self._plotter_widget.render()
            self._hover_dialog.hide()
            return

        # Ignore gray (non-selected) cells when filters active
        if hasattr(self, "_current_mask") and not self._current_mask[idx]:
            self._clear_hover()
            return

        # Build info
        obs_name = str(self._adata.obs_names[idx])
        cluster_val = (
            str(self._adata.obs["clusters"][idx]) if "clusters" in self._adata.obs else "NA"
        )
        source_val = (
            str(self._adata.obs["source"][idx]) if "source" in self._adata.obs else "NA"
        )
        n_counts = (
            str(self._adata.obs["n_counts"][idx]) if "n_counts" in self._adata.obs else "NA"
        )

        gene_info = ""
        if self._current_genes:
            vals = []
            for g in self._current_genes:
                try:
                    val = float(self._adata[idx, g].X)
                except Exception:
                    val = float("nan")
                vals.append(f"{g}:{val:.2f}")
            gene_info = "\n" + ", ".join(vals)

        text = (
            f"Cell: {obs_name}\nCluster: {cluster_val}\nSection: {source_val}\n"
            f"n_counts: {n_counts}{gene_info}"
        )

        # Highlight sphere
        if self._highlight_actor is not None:
            try:
                self._plotter_widget.remove_actor(self._highlight_actor)
            except Exception:
                pass

        cam_pos = self._plotter_widget.camera_position  # preserve

        # Determine sphere radius ~ visual point size
        scene_scale = float(np.max(self._coords_all.ptp(axis=0)))
        base_radius = scene_scale * 0.004  # tuned empirically
        sphere = pv.Sphere(radius=base_radius, center=pick_pt)
        self._highlight_actor = self._plotter_widget.add_mesh(
            sphere,
            color="yellow",
            style="wireframe",
            line_width=2,
            name="hover_sphere",
        )
        self._plotter_widget.render()

        # Restore camera to eliminate any subtle auto-adjustment
        self._plotter_widget.camera_position = cam_pos

        # Show popup near cursor
        global_pos = QCursor.pos() + QPoint(15, 15)
        self._hover_dialog.setText(text)
        self._hover_dialog.adjustSize()
        self._hover_dialog.move(global_pos)
        self._hover_dialog.show()

    # ---------------------------------------------------------------------
    # Gene Only toggle
    # ---------------------------------------------------------------------
    def _toggle_gene_only_mode(self, checked: bool):
        self._gene_only = checked
        # Disable expression threshold only when plotting raw spot CSVs
        self._expr_threshold.setEnabled(not checked)
        if checked:
            self._mode = "gene_spots"
            self._render_gene_only()
        else:
            # revert based on loaded data
            if self._gene_adata is not None:
                self._mode = "gene"
                self._render_gene_mode()
            else:
                self._mode = "cell"
                self._render_spatial()

        # Update summary for non-cell modes as well
        if self._mode != "cell":
            self._update_selection_summary(None)

 