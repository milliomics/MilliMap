import sys
import pathlib
from typing import Optional

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from PyQt5.QtCore import Qt, QMimeData, QPoint
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # for embedded plots
from scipy.spatial import cKDTree  # fast nearest-neighbour lookup
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QCursor, QIcon, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QApplication,
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
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
    QDialog,
    QTabWidget,  # NEW: for vertical tab layout in color dialog
    QScrollArea,
    QFrame,
    QCheckBox,
    QInputDialog,
)
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
import matplotlib.path as mpath
from pyvistaqt import QtInteractor
import pyvista as pv
import vtk
import pandas as pd  # new for CSV loading


class SpatialOmicsViewer(QMainWindow):
    """A lightweight viewer for spatial omics AnnData objects (.h5ad).

    Users can drag-and-drop an AnnData file onto the window or use the
    "Load File" button to pick a file. The viewer looks for the
    ``obsm['spatial']`` matrix (\(n\_cells \times 3\)) plus optional
    ``obs['clusters']`` categorical information to colour points.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spatial Omics Viewer")
        self.resize(1400, 800)
        self.setAcceptDrops(True)

        # -----------------------------------------------------------------
        # Set window icon
        # -----------------------------------------------------------------
        icon_path = "/Users/farah/Library/CloudStorage/GoogleDrive-qianluf2@illinois.edu/My Drive/Milliomics/Designs/cakeinvert.png"
        p = pathlib.Path(icon_path)
        if p.exists():
            pix = QPixmap(str(p))
            # Composite over white background to avoid transparency issues
            white_bg = QPixmap(pix.size())
            white_bg.fill(Qt.white)
            painter = QPainter(white_bg)
            painter.drawPixmap(0, 0, pix)
            painter.end()
            self.setWindowIcon(QIcon(white_bg))

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

        self._side_layout.addWidget(self._gene_group)
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
        ana_poly_btn.clicked.connect(self._start_polygon_selection)
        ana_circle_btn = QPushButton("+ Circle")
        ana_circle_btn.clicked.connect(self._start_circle_selection)
        ana_point_btn = QPushButton("+ Point")
        ana_point_btn.clicked.connect(self._start_point_selection)

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
        self._analysis_btn.toggled.connect(self._toggle_analysis_mode)
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

        # New: Analysis mode flag
        self._analysis_mode = False

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
                    palette = self._generate_plotly_extended_palette(len(all_cats))
                elif self._color_scheme == "custom_turbo":
                    palette = self._generate_custom_turbo_palette(len(all_cats))
                elif self._color_scheme == "sns_palette":
                    palette = self._generate_sns_palette(len(all_cats))
                elif self._color_scheme == "milliomics":
                    palette = self._generate_milliomics_palette(len(all_cats))
                else:  # fallback to plotly_d3
                    palette = self._generate_plotly_extended_palette(len(all_cats))
                
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
            self._plotter_widget.reset_camera()
            self._has_initial_camera = True

        self._plotter_widget.render()

        # Save mask for hover use
        self._current_mask = mask

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
    def _generate_plotly_extended_palette(self, n_colors: int) -> list:
        """Generate extended Plotly/D3 color palette with up to 60 distinct colors."""
        # Original 10 Plotly/D3 colors
        base_colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf'   # cyan
        ]
        
        palette = []
        
        # Convert base colors to RGB
        for color_hex in base_colors:
            rgb = mcolors.to_rgb(color_hex)
            palette.append(rgb)
        
        # If we need more than 10 colors, generate variations
        if n_colors > 10:
            import colorsys
            
            # Generate lighter variants (colors 11-20)
            for color_hex in base_colors:
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Lighter version: increase value, slightly decrease saturation
                v_light = min(1.0, v * 1.3)
                s_light = max(0.3, s * 0.7)
                r_light, g_light, b_light = colorsys.hsv_to_rgb(h, s_light, v_light)
                palette.append((r_light, g_light, b_light))
                if len(palette) >= n_colors:
                    break
        
        # If we need more than 20 colors, generate darker variants (colors 21-30)
        if n_colors > 20:
            for color_hex in base_colors:
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Darker version: decrease value, increase saturation
                v_dark = max(0.3, v * 0.6)
                s_dark = min(1.0, s * 1.2)
                r_dark, g_dark, b_dark = colorsys.hsv_to_rgb(h, s_dark, v_dark)
                palette.append((r_dark, g_dark, b_dark))
                if len(palette) >= n_colors:
                    break
        
        # If we need more than 30 colors, generate desaturated variants (colors 31-40)
        if n_colors > 30:
            for color_hex in base_colors:
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Desaturated version: decrease saturation, maintain value
                s_desat = max(0.2, s * 0.4)
                v_desat = min(1.0, v * 1.1)
                r_desat, g_desat, b_desat = colorsys.hsv_to_rgb(h, s_desat, v_desat)
                palette.append((r_desat, g_desat, b_desat))
                if len(palette) >= n_colors:
                    break
        
        # If we need more than 40 colors, generate hue-shifted variants (colors 41-50)
        if n_colors > 40:
            for color_hex in base_colors:
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Hue-shifted version: shift hue by 30 degrees
                h_shift = (h + 0.083) % 1.0  # 30 degrees = 30/360 = 0.083
                r_shift, g_shift, b_shift = colorsys.hsv_to_rgb(h_shift, s, v)
                palette.append((r_shift, g_shift, b_shift))
                if len(palette) >= n_colors:
                    break
        
        # If we need more than 50 colors, generate high-saturation variants (colors 51-60)
        if n_colors > 50:
            for color_hex in base_colors:
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # High-saturation version: maximize saturation, adjust value
                s_high = 1.0
                v_high = max(0.5, min(1.0, v * 0.9))
                r_high, g_high, b_high = colorsys.hsv_to_rgb(h, s_high, v_high)
                palette.append((r_high, g_high, b_high))
                if len(palette) >= n_colors:
                    break
        
        return palette[:n_colors] if n_colors <= 60 else palette

    def _generate_custom_turbo_palette(self, n_colors: int) -> list:
        """Generate Custom Turbo color palette - vibrant, high-contrast colors."""
        # Use matplotlib's turbo colormap as base
        cmap = plt.get_cmap("turbo", n_colors)
        palette = []
        
        for i in range(n_colors):
            rgb = cmap(i)[:3]  # Get RGB, ignore alpha
            palette.append(rgb)
        
        return palette

    def _generate_sns_palette(self, n_colors: int) -> list:
        """Generate seaborn color palette with multiple beautiful options."""
        # Use different seaborn palettes in sequence for variety
        palette = []
        
        # Base seaborn palettes to cycle through
        sns_palettes = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
        
        colors_per_palette = max(1, n_colors // len(sns_palettes) + 1)
        
        for i, palette_name in enumerate(sns_palettes):
            try:
                pal = sns.color_palette(palette_name, colors_per_palette)
                palette.extend(pal)
                if len(palette) >= n_colors:
                    break
            except:
                # Fallback to default if palette doesn't exist
                pal = sns.color_palette("husl", colors_per_palette)
                palette.extend(pal)
                if len(palette) >= n_colors:
                    break
        
        # If we still need more colors, add husl colors
        if len(palette) < n_colors:
            remaining = n_colors - len(palette)
            extra_colors = sns.color_palette("husl", remaining)
            palette.extend(extra_colors)
        
        return palette[:n_colors]

    def _generate_milliomics_palette(self, n_colors: int) -> list:
        """Generate Milliomics brand color palette using DD596B, 313131, and 6D9F37."""
        # Milliomics brand colors
        brand_colors = [
            "#DD596B",  # Milliomics pink/red
            "#6D9F37",  # Milliomics green
            "#313131",  # Milliomics dark gray
        ]
        
        palette = []
        
        # Convert brand colors to RGB
        for color_hex in brand_colors:
            rgb = mcolors.to_rgb(color_hex)
            palette.append(rgb)
        
        # If we need more colors, generate variations of the brand colors
        if n_colors > 3:
            import colorsys
            
            # Generate lighter variants
            for color_hex in brand_colors:
                if len(palette) >= n_colors:
                    break
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Lighter version
                v_light = min(1.0, v * 1.4)
                s_light = max(0.3, s * 0.8)
                r_light, g_light, b_light = colorsys.hsv_to_rgb(h, s_light, v_light)
                palette.append((r_light, g_light, b_light))
            
            # Generate darker variants
            for color_hex in brand_colors:
                if len(palette) >= n_colors:
                    break
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Darker version
                v_dark = max(0.2, v * 0.6)
                s_dark = min(1.0, s * 1.2)
                r_dark, g_dark, b_dark = colorsys.hsv_to_rgb(h, s_dark, v_dark)
                palette.append((r_dark, g_dark, b_dark))
            
            # Generate desaturated variants
            for color_hex in brand_colors:
                if len(palette) >= n_colors:
                    break
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Desaturated version
                s_desat = max(0.2, s * 0.5)
                v_desat = min(1.0, v * 1.1)
                r_desat, g_desat, b_desat = colorsys.hsv_to_rgb(h, s_desat, v_desat)
                palette.append((r_desat, g_desat, b_desat))
            
            # If still need more, generate hue-shifted variants
            while len(palette) < n_colors:
                for color_hex in brand_colors:
                    if len(palette) >= n_colors:
                        break
                    r, g, b = mcolors.to_rgb(color_hex)
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    # Hue-shifted version
                    h_shift = (h + 0.15) % 1.0  # 54 degrees shift
                    r_shift, g_shift, b_shift = colorsys.hsv_to_rgb(h_shift, s, v)
                    palette.append((r_shift, g_shift, b_shift))
        
        return palette[:n_colors]

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

        # Populate clusters
        if "clusters" in self._adata.obs:
            for cl in np.sort(self._adata.obs["clusters"].astype(str).unique()):
                item = QListWidgetItem(str(cl))
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
        self._current_clusters = [itm.text() for itm in self._cluster_list.selectedItems()]
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

    def _perform_gene_search(self) -> None:
        self._filter_list_widget(self._gene_list, self._gene_search_line.text())

    def _perform_section_search(self) -> None:
        self._filter_list_widget(self._section_list, self._section_search_line.text())

    # ---------------------------------------------------------------------
    # Orientation axes
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Analysis mode helpers
    # ---------------------------------------------------------------------
    def _toggle_analysis_mode(self, checked: bool):
        """Enable/disable analysis mode and show/hide tools."""
        self._analysis_mode = checked
        self._analysis_group.setVisible(checked)
        if not checked:
            # Disable any active picking
            try:
                self._plotter_widget.disable_picking()
            except Exception:
                pass
            # Remove selection actor
            if self._analysis_selection_actor is not None:
                try:
                    self._plotter_widget.remove_actor(self._analysis_selection_actor)
                except Exception:
                    pass
                self._analysis_selection_actor = None
            self._plotter_widget.render()

    def _start_polygon_selection(self):
        """Ask user for sections, then open 2-D lasso selection window."""
        if self._adata is None or self._coords_all is None:
            QMessageBox.warning(self, "No Data", "Load data and render first.")
            return

        # Collect section names
        sources = []
        if "source" in self._adata.obs:
            sources = sorted(self._adata.obs["source"].astype(str).unique())

        dlg = SectionSelectDialog(sources, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        selected_srcs = dlg.get_selected_sources()
        self._open_polygon_window(selected_srcs)

    def _open_polygon_window(self, selected_srcs: list[str]):
        """Open a separate dialog with 2-D scatter and lasso selector."""
        # Determine indices to include
        mask = np.ones(self._coords_all.shape[0], dtype=bool)
        if selected_srcs and "source" in self._adata.obs:
            mask = self._adata.obs["source"].astype(str).isin(selected_srcs).values

        coords2d = self._coords_all[mask][:, :2]  # drop z
        indices = np.where(mask)[0]

        # Cluster labels for color coding
        clusters = None
        if "clusters" in self._adata.obs:
            clusters = self._adata.obs.iloc[indices]["clusters"].astype(str).to_numpy()

        if coords2d.shape[0] == 0:
            QMessageBox.information(self, "No Cells", "No cells in the selected section(s).")
            return

        dialog = LassoSelectionDialog(coords2d, indices, clusters, self)
        dialog.exec_()

    def _start_circle_selection(self):
        if self._adata is None or self._coords_all is None:
            QMessageBox.warning(self, "No Data", "Load data and render first.")
            return

        # Disable any previous picking to avoid PyVista error
        try:
            self._plotter_widget.disable_picking()
        except Exception:
            pass

        def _picked(picked_point):
            if picked_point is None or len(picked_point) == 0:
                return
            center = np.array(picked_point)
            # Ask for radius
            rad, ok = QInputDialog.getDouble(self, "Circle Radius", "Enter radius (same units as coords):", 50.0, 0.1, 1e6, 1)
            if not ok:
                return
            # Query KD-tree or direct distance
            if self._kd_tree is not None:
                idxs = self._kd_tree.query_ball_point(center, r=rad)
            else:
                dists = np.linalg.norm(self._coords_all - center, axis=1)
                idxs = np.where(dists <= rad)[0].tolist()
            if not idxs:
                QMessageBox.information(self, "Selection", "No cells within radius.")
                return
            self._process_selection(self._coords_all[idxs])

        # Enable single point picking
        self._plotter_widget.enable_point_picking(callback=lambda mesh: _picked(mesh.points[0]), show_message=True)

    def _start_point_selection(self):
        if self._adata is None or self._coords_all is None:
            QMessageBox.warning(self, "No Data", "Load data and render first.")
            return

        try:
            self._plotter_widget.disable_picking()
        except Exception:
            pass

        def _picked(mesh):
            if mesh is None or mesh.n_points == 0:
                return
            pt = mesh.points[0]
            self._process_selection(np.array([pt]))

        self._plotter_widget.enable_point_picking(callback=_picked, show_message=True)

    def _process_selection(self, selected_pts: np.ndarray):
        """Given a set of selected XYZ coordinates, compute stats and plot."""
        if self._coords_all is None:
            return

        # Match selected coordinates back to indices – use tolerance for float comparisons
        sel_indices = []
        for pt in selected_pts:
            # Compare with tolerance 1e-5
            dists = np.linalg.norm(self._coords_all - pt, axis=1)
            idx = np.where(dists < 1e-5)[0]
            if idx.size:
                sel_indices.extend(idx.tolist())
        if not sel_indices:
            QMessageBox.information(self, "Selection", "No cells matched the selected region.")
            return

        sel_indices = np.unique(sel_indices)

        if "clusters" not in self._adata.obs:
            QMessageBox.information(self, "Missing Clusters", "AnnData lacks 'clusters' information for analysis.")
            return
        clusters = self._adata.obs.iloc[sel_indices]["clusters"].astype(str)
        counts = clusters.value_counts()
        perc = counts / counts.sum() * 100

        # Plot percentages in embedded dialog ---------------------------------
        dlg = QDialog(self)
        dlg.setWindowTitle("Cell type composition")
        dlg.resize(500, 400)
        vbox = QVBoxLayout(dlg)
        canvas = FigureCanvas(plt.Figure(figsize=(5, 3)))
        vbox.addWidget(canvas)
        ax = canvas.figure.subplots()
        ax.bar(perc.index, perc.values, color="#1f77b4")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Cell type composition in selection")
        ax.set_xticklabels(perc.index, rotation=45, ha="right")
        canvas.figure.tight_layout()
        dlg.exec_()

        # Highlight selected region in viewer (optional)
        if self._analysis_selection_actor is not None:
            try:
                self._plotter_widget.remove_actor(self._analysis_selection_actor)
            except Exception:
                pass
        self._analysis_selection_actor = self._plotter_widget.add_mesh(
            pv.PolyData(selected_pts),
            color="yellow",
            opacity=0.3,
            point_size=8,
            render_points_as_spheres=True,
        )
        self._plotter_widget.render()


# -----------------------------------------------------------------------------
# Color scheme selection dialog
# -----------------------------------------------------------------------------


class ColorSchemeDialog(QDialog):
    """Dialog for selecting color schemes."""

    def __init__(self, parent=None, current_scheme="plotly_d3", adata=None):
        super().__init__(parent)
        self.setWindowTitle("Color Scheme Selection")
        self.setFixedSize(400, 500)
        self._current_scheme = current_scheme
        self._adata = adata
        self._selected_scheme = current_scheme
        
        self._setup_ui()

    def _setup_ui(self):
        # Completely redesigned UI: vertical tabs on the left + description on the right
        from PyQt5.QtWidgets import QSizePolicy  # local import to avoid header change noise

        # Reset layout (ensure we start clean)
        self.setLayout(QVBoxLayout())
        main_layout: QVBoxLayout = self.layout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ------------------------------------------------------------------
        # Vertical tabs -----------------------------------------------------
        # ------------------------------------------------------------------
        self._tabs = QTabWidget()
        # Horizontal tabs across the top; enable scroll arrows when space is limited
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.tabBar().setUsesScrollButtons(True)
        self._tabs.setMovable(False)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self._tabs, 1)

        # Map of scheme key → button (for styling)
        self._scheme_buttons: dict[str, QPushButton] = {}

        # Helper ----------------------------------------------------------------
        def _make_page(title: str, lines: list[str], scheme_key: str, enabled: bool = True) -> QWidget:
            page = QWidget()
            vbox = QVBoxLayout(page)
            vbox.setContentsMargins(10, 10, 10, 10)
            vbox.setSpacing(6)
        
            lbl_title = QLabel(title)
            lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")
            vbox.addWidget(lbl_title)

            for ln in lines:
                vbox.addWidget(QLabel(ln))

            vbox.addStretch()

            btn = QPushButton(f"Use {title} Colors")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _=None, sk=scheme_key: self._select_scheme(sk))
            btn.setEnabled(enabled)
            vbox.addWidget(btn)

            # Store for styling updates
            self._scheme_buttons[scheme_key] = btn
            # Also keep the old attribute names so other methods continue to work
            setattr(self, f"_{scheme_key}_btn", btn)
            return page

        # AnnData tab ----------------------------------------------------------
        adata_enabled = self._adata is not None and "clusters_colors" in getattr(self._adata, "obs", {})
        adata_lines = (
            ["✅ Use colors from loaded AnnData file", "Uses the original cluster colors defined in your data."]
            if adata_enabled
            else [
                "⚠️ AnnData file does not contain 'clusters_colors'",
                "Load an AnnData file with cluster colors to use this option.",
            ]
        )
        self._tabs.addTab(_make_page("AnnData", adata_lines, "anndata", enabled=adata_enabled), "AnnData")

        # Plotly/D3 tab --------------------------------------------------------
        self._tabs.addTab(
            _make_page(
                "Plotly/D3",
                [
                    "🎨 Professional Plotly/D3 color palette",
                    "• Up to 60 distinct colors",
                    "• Industry standard colors",
                    "• Optimized for data visualization",
                ],
                "plotly_d3",
            ),
            "Plotly/D3",
        )

        # Custom Turbo tab -----------------------------------------------------
        self._tabs.addTab(
            _make_page(
                "Custom Turbo",
                [
                    "🌈 Custom Turbo color palette",
                    "• Vibrant, high-contrast colors",
                    "• Based on matplotlib's turbo colormap",
                    "• Excellent for many clusters",
                ],
                "custom_turbo",
            ),
            "Custom Turbo",
        )

        # SNS Palette tab ------------------------------------------------------
        self._tabs.addTab(
            _make_page(
                "SNS Palettes",
                [
                    "📊 Seaborn color palettes",
                    "• Multiple beautiful seaborn palettes",
                    "• Scientifically optimized colors",
                    "• Perfect for statistical visualization",
                ],
                "sns_palette",
            ),
            "SNS",
        )

        # Milliomics tab -------------------------------------------------------
        self._tabs.addTab(
            _make_page(
                "Milliomics",
                [
                    "🎂 Milliomics brand colors",
                    "• Custom Milliomics color scheme",
                    "• Pink (#DD596B), Green (#6D9F37), Gray (#313131)",
                    "• Brand-consistent visualization",
                ],
                "milliomics",
            ),
            "Milliomics",
        )

        # ------------------------------------------------------------------
        # Bottom OK / Cancel ------------------------------------------------
        # ------------------------------------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        self._ok_btn = QPushButton("OK")
        self._ok_btn.clicked.connect(self.accept)
        bottom_row.addWidget(self._cancel_btn)
        bottom_row.addSpacing(10)
        bottom_row.addWidget(self._ok_btn)
        main_layout.addLayout(bottom_row)
        
        # Highlight the current scheme --------------------------------------
        idx_lookup = {"anndata": 0, "plotly_d3": 1, "custom_turbo": 2, "sns_palette": 3, "milliomics": 4}
        self._tabs.setCurrentIndex(idx_lookup.get(self._current_scheme, 1))
        self._select_scheme(self._current_scheme)

    def _on_tab_changed(self, index: int):
        key_map = {0: "anndata", 1: "plotly_d3", 2: "custom_turbo", 3: "sns_palette", 4: "milliomics"}
        scheme = key_map.get(index)
        if scheme:
            self._select_scheme(scheme)

    def _select_scheme(self, scheme: str):
        """Store chosen scheme and refresh button styles."""
        self._selected_scheme = scheme
        self._update_button_styles()

    def _update_button_styles(self):
        # Reset styles for all scheme buttons
        for btn in self._scheme_buttons.values():
            btn.setStyleSheet("")

        # Highlight selected button with a filled background (no overlay triangle)
        scheme_colors = {
            "anndata": "#666666",
            "plotly_d3": "#1f77b4",
            "custom_turbo": "#ff4e00",
            "sns_palette": "#4c72b0",
            "milliomics": "#DD596B",
        }
        sel_btn = self._scheme_buttons.get(self._selected_scheme)
        if sel_btn is None:
            return

        col = scheme_colors.get(self._selected_scheme, "#DD596B")
        sel_style = f"""
            QPushButton {{
                background-color: {col};
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {col};
            }}
        """
        sel_btn.setStyleSheet(sel_style)

    def get_selected_scheme(self):
        return self._selected_scheme


# -----------------------------------------------------------------------------
# Small popup dialog for hover info
# -----------------------------------------------------------------------------


class HoverInfoDialog(QLabel):
    """Tooltip-like label to display cell info."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip)
        self.setStyleSheet(
            "QLabel {background-color: rgba(0,0,0,0.75); color: white; border: 1px solid white; padding: 4px;}"
        )
        self.hide()


# -----------------------------------------------------------------------------
# Dialogs for polygon selection workflow
# -----------------------------------------------------------------------------


class SectionSelectDialog(QDialog):
    """Dialog allowing user to choose which sections (sources) to include."""

    def __init__(self, sources: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Sections")
        self.resize(300, 400)

        layout = QVBoxLayout(self)

        self._all_chk = QCheckBox("All sections")
        self._all_chk.setChecked(True)
        layout.addWidget(self._all_chk)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); vbox = QVBoxLayout(content)
        self._check_boxes: list[QCheckBox] = []
        for src in sources:
            cb = QCheckBox(src)
            cb.setChecked(True)
            self._check_boxes.append(cb)
            vbox.addWidget(cb)
        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Connect master checkbox to children
        self._all_chk.stateChanged.connect(lambda state: [cb.setChecked(state == Qt.Checked) for cb in self._check_boxes])

        # Ensure master checkbox reflects children states
        def _child_changed(_):
            all_on = all(cb.isChecked() for cb in self._check_boxes)
            self._all_chk.blockSignals(True)
            self._all_chk.setChecked(all_on)
            self._all_chk.blockSignals(False)

        for cb in self._check_boxes:
            cb.stateChanged.connect(_child_changed)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel"); cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK"); ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def get_selected_sources(self) -> list[str]:
        if self._all_chk.isChecked():
            return []  # empty means all
        return [cb.text() for cb in self._check_boxes if cb.isChecked()]


class LassoSelectionDialog(QDialog):
    """Dialog showing 2-D scatter and allowing polygon selection with pan/zoom."""

    def __init__(self, coords2d: np.ndarray, indices: np.ndarray, clusters: Optional[np.ndarray], viewer: "SpatialOmicsViewer"):
        super().__init__(viewer)
        self.setWindowTitle("Polygon Selection (2-D)")
        self.resize(700, 700)

        self._coords2d = coords2d
        self._indices = indices
        self._viewer = viewer

        vbox = QVBoxLayout(self)
        fig = Figure(facecolor="black")
        self._canvas = FigureCanvas(fig)
        vbox.addWidget(self._canvas)
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        vbox.addWidget(NavigationToolbar2QT(self._canvas, self))

        self._ax = fig.add_subplot(111)

        # Color-coding by clusters if provided
        if clusters is not None:
            uniq = np.unique(clusters)
            # Use viewer's palette util if available else tab20
            if hasattr(viewer, "_generate_plotly_extended_palette"):
                palette = viewer._generate_plotly_extended_palette(len(uniq))
            else:
                cmap = plt.get_cmap("tab20", len(uniq))
                palette = [cmap(i)[:3] for i in range(len(uniq))]
            lookup = {u: palette[i] for i, u in enumerate(uniq)}
            colors = [lookup[c] for c in clusters]
        else:
            colors = "white"

        self._ax.scatter(coords2d[:, 0], coords2d[:, 1], s=6, c=colors)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_facecolor("black")
        self._ax.set_xticks([]); self._ax.set_yticks([])

        # Polygon selector (click to add points, double-click/enter to finish)
        # Older Matplotlib versions (<3.8) don't support the "interactive" keyword.
        self._poly = PolygonSelector(
            self._ax,
            self._on_select,
            props=dict(color="yellow", linewidth=2, alpha=0.8),
        )

        instruct = QLabel("Click to add vertices. Double-click or press Enter to finish.\nShift+Drag: pan, Scroll: zoom")
        instruct.setStyleSheet("color: white;")
        vbox.addWidget(instruct)

        # Pan/zoom handlers via events
        self._press_event = None
        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_scroll = self._canvas.mpl_connect("scroll_event", self._on_scroll)

    # ---------------- selection ----------------
    def _on_select(self, verts):
        path = mpath.Path(verts)
        mask = path.contains_points(self._coords2d)
        if not mask.any():
            QMessageBox.information(self, "Selection", "No cells inside polygon.")
            return
        selected_idxs = self._indices[mask]
        self._viewer._process_selection(self._viewer._coords_all[selected_idxs])
        self.accept()

    # ---------------- pan/zoom ----------------
    def _on_press(self, event):
        if event.key == "shift" and event.button == 1:
            self._press_event = event

    def _on_release(self, event):
        self._press_event = None

    def _on_motion(self, event):
        if self._press_event is None:
            return
        dx = event.xdata - self._press_event.xdata if event.xdata and self._press_event.xdata else 0
        dy = event.ydata - self._press_event.ydata if event.ydata and self._press_event.ydata else 0
        if dx == 0 and dy == 0:
            return
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        self._ax.set_xlim(xlim[0]-dx, xlim[1]-dx)
        self._ax.set_ylim(ylim[0]-dy, ylim[1]-dy)
        self._canvas.draw_idle()

    def _on_scroll(self, event):
        base_scale = 1.2
        if event.button == "up":
            scale = 1/base_scale
        elif event.button == "down":
            scale = base_scale
        else:
            scale = 1
        xlim = self._ax.get_xlim(); ylim = self._ax.get_ylim()
        cur_xrange = (xlim[1]-xlim[0]) * scale
        cur_yrange = (ylim[1]-ylim[0]) * scale
        x_center = event.xdata if event.xdata is not None else (xlim[0]+xlim[1])/2
        y_center = event.ydata if event.ydata is not None else (ylim[0]+ylim[1])/2
        self._ax.set_xlim(x_center - cur_xrange/2, x_center + cur_xrange/2)
        self._ax.set_ylim(y_center - cur_yrange/2, y_center + cur_yrange/2)
        self._canvas.draw_idle()


def main() -> None:
    app = QApplication(sys.argv)
    # Set application icon globally
    icon_path = "/Users/farah/Library/CloudStorage/GoogleDrive-qianluf2@illinois.edu/My Drive/Milliomics/Designs/cakeinvert.png"
    if pathlib.Path(icon_path).exists():
        app.setWindowIcon(QIcon(icon_path))
    viewer = SpatialOmicsViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 