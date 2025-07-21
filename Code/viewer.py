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

 