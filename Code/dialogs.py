"""Dialog components for spatial omics viewer."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.widgets import PolygonSelector
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
    QWidget, QScrollArea, QCheckBox, QMessageBox, QInputDialog, QSizePolicy
)
from typing import Optional, List

# Handle both direct execution and package import
try:
    from .colors import (
        generate_plotly_extended_palette,
        generate_custom_turbo_palette, 
        generate_sns_palette,
        generate_milliomics_palette
    )
except (ImportError, ValueError):
    from colors import (
        generate_plotly_extended_palette,
        generate_custom_turbo_palette, 
        generate_sns_palette,
        generate_milliomics_palette
    )


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


class HoverInfoDialog(QLabel):
    """Tooltip-like label to display cell info."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip)
        self.setStyleSheet(
            "QLabel {background-color: rgba(0,0,0,0.75); color: white; border: 1px solid white; padding: 4px;}"
        )
        self.hide()


class SectionSelectDialog(QDialog):
    """Dialog allowing user to choose which sections (sources) to include."""

    def __init__(self, sources: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Sections")
        self.resize(300, 400)

        layout = QVBoxLayout(self)

        self._all_chk = QCheckBox("All sections")
        self._all_chk.setChecked(True)
        layout.addWidget(self._all_chk)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); vbox = QVBoxLayout(content)
        self._check_boxes: List[QCheckBox] = []
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

    def get_selected_sources(self) -> List[str]:
        if self._all_chk.isChecked():
            return []  # empty means all
        return [cb.text() for cb in self._check_boxes if cb.isChecked()]


class LassoSelectionDialog(QDialog):
    """Dialog showing 2-D scatter and allowing polygon selection with pan/zoom."""

    def __init__(self, coords2d: np.ndarray, indices: np.ndarray, clusters: Optional[np.ndarray], viewer):
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