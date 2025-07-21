"""Dialog components for spatial omics viewer."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.widgets import PolygonSelector
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget,
    QWidget, QScrollArea, QCheckBox, QMessageBox, QInputDialog, QSizePolicy,
    QFrame
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
        self.setWindowTitle("🔺 Millimap - Polygon Selection (2-D)")
        self.resize(700, 700)
        
        # Modern dialog styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: white;
            }
        """)

        self._coords2d = coords2d
        self._indices = indices
        self._viewer = viewer

        vbox = QVBoxLayout(self)
        fig = Figure(facecolor="black")
        self._canvas = FigureCanvas(fig)
        vbox.addWidget(self._canvas)
        
        # Create custom modern toolbar instead of default matplotlib toolbar
        self._create_custom_toolbar(vbox)

        self._ax = fig.add_subplot(111)

        # Color-coding by clusters using viewer's color scheme
        if clusters is not None:
            uniq = np.unique(clusters)
            
            # Use viewer's color scheme exactly like main screen
            if hasattr(viewer, '_color_scheme'):
                color_scheme = viewer._color_scheme
            else:
                color_scheme = "plotly_d3"  # fallback
            
            # Generate colors using same logic as main screen
            if color_scheme == "anndata" and hasattr(viewer, '_adata') and viewer._adata is not None:
                if "clusters_colors" in viewer._adata.obs:
                    try:
                        # Create a mapping from cluster to color from the full dataset
                        all_clusters = viewer._adata.obs["clusters"].astype(str)
                        all_colors = viewer._adata.obs["clusters_colors"]
                        color_map = dict(zip(all_clusters, all_colors))
                        # Map the colors for our subset
                        colors = [mcolors.to_rgb(str(color_map.get(c, 'white'))) for c in clusters]
                    except Exception:
                        # Fallback if color parsing fails
                        colors = "white"
                else:
                    colors = "white"
            else:
                # Import the color functions (they should be available from viewer)
                try:
                    if color_scheme == "plotly_d3":
                        from .colors import generate_plotly_extended_palette
                        palette = generate_plotly_extended_palette(len(uniq))
                    elif color_scheme == "custom_turbo":
                        from .colors import generate_custom_turbo_palette
                        palette = generate_custom_turbo_palette(len(uniq))
                    elif color_scheme == "sns_palette":
                        from .colors import generate_sns_palette
                        palette = generate_sns_palette(len(uniq))
                    elif color_scheme == "milliomics":
                        from .colors import generate_milliomics_palette
                        palette = generate_milliomics_palette(len(uniq))
                    else:  # fallback to plotly_d3
                        from .colors import generate_plotly_extended_palette
                        palette = generate_plotly_extended_palette(len(uniq))
                    
                    lookup = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
                    colors = [lookup[c] for c in clusters]
                except (ImportError, ValueError):
                    # Fallback for import issues
                    try:
                        if hasattr(viewer, "_generate_plotly_extended_palette"):
                            palette = viewer._generate_plotly_extended_palette(len(uniq))
                        else:
                            cmap = plt.get_cmap("tab20", len(uniq))
                            palette = [cmap(i)[:3] for i in range(len(uniq))]
                        lookup = {u: palette[i] for i, u in enumerate(uniq)}
                        colors = [lookup[c] for c in clusters]
                    except Exception:
                        colors = "white"
        else:
            colors = "white"

        # Use same spot size as main screen (5)
        self._ax.scatter(coords2d[:, 0], coords2d[:, 1], s=5, c=colors)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_facecolor("black")
        self._ax.set_xticks([]); self._ax.set_yticks([])

        # Initialize polygon mode state
        self._polygon_mode = False
        self._poly = None
        self._shift_pressed = False
        
        # Initialize polygon selector (will be created when polygon mode is activated)
        self._create_polygon_selector()

        instruct = QLabel("🎯 Pan & Zoom mode: Shift+drag to pan • Mouse wheel to zoom • Click 'Add Polygon' button to start drawing")
        instruct.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                background-color: #333;
                padding: 12px;
                border-radius: 8px;
                font-size: 13px;
                margin: 4px;
                border: 1px solid #555;
            }
        """)
        vbox.addWidget(instruct)

        # Pan/zoom handlers via events
        self._press_event = None
        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_scroll = self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._cid_key_press = self._canvas.mpl_connect("key_press_event", self._on_key_press)
        self._cid_key_release = self._canvas.mpl_connect("key_release_event", self._on_key_release)
        
        # Make sure the canvas can receive focus for key events
        self._canvas.setFocusPolicy(Qt.ClickFocus)
        self._canvas.setFocus()
        
        # Initialize view history for back/forward navigation
        self._view_history = []
        self._view_position = -1
        self._save_view()  # Save initial view

    def _create_custom_toolbar(self, parent_layout):
        """Create a modern, styled toolbar for navigation."""
        # Toolbar container with dark styling
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 8px;
                margin: 2px;
            }
        """)
        
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setSpacing(8)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        
        # Button styling for modern appearance
        button_style = """
            QPushButton {
                background-color: #404040;
                color: white;
                border: 2px solid #606060;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #707070;
            }
            QPushButton:pressed {
                background-color: #353535;
                border-color: #505050;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
                border-color: #444;
            }
        """
        
        # Home button
        home_btn = QPushButton("🏠 Home")
        home_btn.setStyleSheet(button_style)
        home_btn.clicked.connect(self._home_view)
        home_btn.setToolTip("Reset view to show all data")
        toolbar_layout.addWidget(home_btn)
        
        # Back button  
        back_btn = QPushButton("⬅ Back")
        back_btn.setStyleSheet(button_style)
        back_btn.clicked.connect(self._back_view)
        back_btn.setToolTip("Go back to previous view")
        toolbar_layout.addWidget(back_btn)
        
        # Forward button
        forward_btn = QPushButton("➡ Forward")
        forward_btn.setStyleSheet(button_style)
        forward_btn.clicked.connect(self._forward_view)
        forward_btn.setToolTip("Go forward to next view")
        toolbar_layout.addWidget(forward_btn)
        
        # Separator
        toolbar_layout.addWidget(self._create_separator())
        
        # Polygon Mode Toggle Button
        self._polygon_btn = QPushButton("🔺 Add Polygon")
        polygon_style = button_style + """
            QPushButton:checked {
                background-color: #2196F3;
                border-color: #1976D2;
            }
        """
        self._polygon_btn.setStyleSheet(polygon_style)
        self._polygon_btn.setCheckable(True)
        self._polygon_btn.toggled.connect(self._toggle_polygon_mode)
        self._polygon_btn.setToolTip("Toggle polygon drawing mode • ESC to exit polygon mode")
        toolbar_layout.addWidget(self._polygon_btn)
        
        # Pan button (for visual feedback only)
        pan_btn = QPushButton("✋ Pan")
        pan_btn.setStyleSheet(button_style)
        pan_btn.setToolTip("Pan: Hold Shift + Left Click and drag")
        pan_btn.setEnabled(False)  # Just for visual reference
        toolbar_layout.addWidget(pan_btn)
        
        # Zoom button (for visual feedback only)
        zoom_btn = QPushButton("🔍 Zoom")
        zoom_btn.setStyleSheet(button_style)
        zoom_btn.setToolTip("Zoom: Mouse wheel to zoom in/out")
        zoom_btn.setEnabled(False)  # Just for visual reference
        toolbar_layout.addWidget(zoom_btn)
        
        # Separator
        toolbar_layout.addWidget(self._create_separator())
        
        # Configure button
        config_btn = QPushButton("⚙️ Configure")
        config_btn.setStyleSheet(button_style)
        config_btn.clicked.connect(self._configure_plot)
        config_btn.setToolTip("Configure plot settings")
        toolbar_layout.addWidget(config_btn)
        
        # Save button
        save_btn = QPushButton("💾 Save")
        save_btn.setStyleSheet(button_style)
        save_btn.clicked.connect(self._save_plot)
        save_btn.setToolTip("Save the current plot")
        toolbar_layout.addWidget(save_btn)
        
        # Stretch to push coordinates to the right
        toolbar_layout.addStretch()
        
        # Coordinates label
        self._coords_label = QLabel("x=0, y=0")
        self._coords_label.setStyleSheet("""
            QLabel {
                color: white;
                font-family: monospace;
                font-size: 11px;
                padding: 4px 8px;
                background-color: #333;
                border-radius: 4px;
            }
        """)
        toolbar_layout.addWidget(self._coords_label)
        
        parent_layout.addWidget(toolbar_frame)
        
        # Connect mouse move to update coordinates
        self._canvas.mpl_connect("motion_notify_event", self._update_coordinates)
        
    def _create_separator(self):
        """Create a visual separator for the toolbar."""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #666; margin: 0px 4px;")
        return separator
        
    def _update_coordinates(self, event):
        """Update coordinate display."""
        if event.inaxes:
            self._coords_label.setText(f"x={event.xdata:.0f}, y={event.ydata:.0f}")
        else:
            self._coords_label.setText("x=---, y=---")
    
    def _home_view(self):
        """Reset view to show all data."""
        self._ax.autoscale()
        self._canvas.draw()
        self._save_view()
        
    def _back_view(self):
        """Go back to previous view."""
        if self._view_position > 0:
            self._view_position -= 1
            self._restore_view(self._view_history[self._view_position])
            
    def _forward_view(self):
        """Go forward to next view."""
        if self._view_position < len(self._view_history) - 1:
            self._view_position += 1
            self._restore_view(self._view_history[self._view_position])
    
    def _save_view(self):
        """Save current view to history."""
        current_view = (self._ax.get_xlim(), self._ax.get_ylim())
        if self._view_position == len(self._view_history) - 1:
            self._view_history.append(current_view)
            self._view_position += 1
        else:
            # Remove forward history when saving new view
            self._view_history = self._view_history[:self._view_position + 1]
            self._view_history.append(current_view)
            self._view_position += 1
            
    def _restore_view(self, view):
        """Restore a saved view."""
        xlim, ylim = view
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        self._canvas.draw()
        
    # ---------------- pan/zoom ----------------
    def _on_press(self, event):
        """Handle mouse press events."""
        # Always handle pan on shift+click, regardless of polygon mode
        if event.key == "shift" and event.button == 1:
            self._press_event = event

    def _on_release(self, event):
        """Handle mouse release events."""
        self._press_event = None

    def _on_motion(self, event):
        """Handle mouse motion events for panning."""
        # Pan functionality (shift+drag)
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
        # Save view for history
        self._save_view()

    def _on_scroll(self, event):
        """Handle mouse scroll events for zooming with improved sensitivity."""
        if event.inaxes != self._ax:
            return
            
        # Use even more gradual zoom for smoother experience
        base_scale = 1.08  # Further reduced for extra smooth zooming
        if event.button == "up":
            scale = 1/base_scale
        elif event.button == "down":
            scale = base_scale
        else:
            scale = 1
            
        # Get current axis limits
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        
        # Calculate new ranges
        cur_xrange = (xlim[1] - xlim[0]) * scale
        cur_yrange = (ylim[1] - ylim[0]) * scale
        
        # Use mouse position as zoom center if available, otherwise use plot center
        if event.xdata is not None and event.ydata is not None:
            x_center = event.xdata
            y_center = event.ydata
        else:
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
        
        # Set new limits with smooth transition
        new_xlim = (x_center - cur_xrange/2, x_center + cur_xrange/2)
        new_ylim = (y_center - cur_yrange/2, y_center + cur_yrange/2)
        
        self._ax.set_xlim(new_xlim)
        self._ax.set_ylim(new_ylim)
        self._canvas.draw_idle()
        # Save view for history
        self._save_view()

    def _toggle_pan(self, checked):
        """Toggle pan mode (handled automatically via shift+drag)."""
        pass
        
    def _toggle_zoom(self, checked):
        """Toggle zoom mode (handled automatically via mouse wheel)."""
        pass
        
    def _configure_plot(self):
        """Open plot configuration dialog."""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Configure", 
                              "Plot configuration:\n\n"
                              "• Pan: Hold Shift + Left Click and drag\n"
                              "• Zoom: Use mouse wheel\n"
                              "• Polygon: Click to add vertices\n"
                              "• Finish: Double-click or press Enter")
        
    def _save_plot(self):
        """Save the current plot to file."""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "polygon_selection.png", 
            "PNG files (*.png);;PDF files (*.pdf);;All files (*.*)"
        )
        if filename:
            self._canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{filename}")

    # ---------------- polygon mode management ----------------
    def _create_polygon_selector(self):
        """Create or recreate the polygon selector."""
        if self._poly is not None:
            self._poly.disconnect_events()
        
        if self._polygon_mode:
            # Create polygon selector only when in polygon mode
            self._poly = PolygonSelector(
                self._ax,
                self._on_select,
                props=dict(color="yellow", linewidth=3, alpha=0.8),
                useblit=True
            )
        else:
            self._poly = None
    
    def _toggle_polygon_mode(self, checked):
        """Toggle polygon drawing mode."""
        self._polygon_mode = checked
        self._create_polygon_selector()
        
        if checked:
            self._polygon_btn.setText("🔺 Exit Polygon")
            self._polygon_btn.setToolTip("Click to add vertices • Double-click to finish • ESC to exit")
            # Update instructions
            self._update_instructions("🔺 POLYGON MODE: Click to add vertices • Double-click or Enter to finish • Shift+drag to pan")
        else:
            self._polygon_btn.setText("🔺 Add Polygon") 
            self._polygon_btn.setToolTip("Toggle polygon drawing mode • ESC to exit polygon mode")
            # Clear any active polygon
            if self._poly is not None:
                self._poly.disconnect_events()
                self._poly = None
            self._canvas.draw()
            self._update_instructions("🎯 Pan & Zoom mode: Shift+drag to pan • Mouse wheel to zoom • Click polygon button to draw")
    
    def _update_instructions(self, text):
        """Update the instruction text dynamically."""
        # Find the instruction label and update it
        for child in self.findChildren(QLabel):
            if "Instructions:" in child.text() or "POLYGON MODE:" in child.text() or "Pan & Zoom mode:" in child.text():
                child.setText(text)
                break
    
    # ---------------- key event handling ----------------
    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'shift':
            self._shift_pressed = True
        elif event.key == 'escape':
            # ESC key exits polygon mode
            if self._polygon_mode:
                self._polygon_btn.setChecked(False)
                self._toggle_polygon_mode(False)
    
    def _on_key_release(self, event):
        """Handle key release events."""
        if event.key == 'shift':
            self._shift_pressed = False
    
    # ---------------- selection ----------------
    def _on_select(self, verts):
        path = mpath.Path(verts)
        mask = path.contains_points(self._coords2d)
        if not mask.any():
            QMessageBox.information(self, "Selection", "No cells inside polygon.")
            return
        selected_idxs = self._indices[mask]
        self._viewer._analysis_tools.process_selection(self._viewer._coords_all[selected_idxs])
        self.accept() 