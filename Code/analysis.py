"""Analysis tools for spatial omics data exploration."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog
)
from scipy.spatial import cKDTree
import pyvista as pv

# Handle both direct execution and package import
try:
    from .dialogs import SectionSelectDialog, LassoSelectionDialog
except (ImportError, ValueError):
    from dialogs import SectionSelectDialog, LassoSelectionDialog


class AnalysisTools:
    """Analysis tools for spatial omics data exploration and region selection."""
    
    def __init__(self, viewer):
        """Initialize analysis tools with reference to main viewer.
        
        Args:
            viewer: Reference to the main SpatialOmicsViewer instance
        """
        self.viewer = viewer
        self._analysis_selection_actor = None
        
    def toggle_analysis_mode(self, checked: bool):
        """Enable/disable analysis mode and show/hide tools."""
        self.viewer._analysis_mode = checked
        self.viewer._analysis_group.setVisible(checked)
        if not checked:
            # Disable any active picking
            try:
                self.viewer._plotter_widget.disable_picking()
            except Exception:
                pass
            # Remove selection actor
            if self._analysis_selection_actor is not None:
                try:
                    self.viewer._plotter_widget.remove_actor(self._analysis_selection_actor)
                except Exception:
                    pass
                self._analysis_selection_actor = None
            self.viewer._plotter_widget.render()

    def start_polygon_selection(self):
        """Ask user for sections, then open 2-D lasso selection window."""
        if self.viewer._adata is None or self.viewer._coords_all is None:
            QMessageBox.warning(self.viewer, "No Data", "Load data and render first.")
            return

        # Collect section names
        sources = []
        if "source" in self.viewer._adata.obs:
            sources = sorted(self.viewer._adata.obs["source"].astype(str).unique())

        dlg = SectionSelectDialog(sources, self.viewer)
        if dlg.exec_() != QDialog.Accepted:
            return

        selected_srcs = dlg.get_selected_sources()
        self._open_polygon_window(selected_srcs)

    def _open_polygon_window(self, selected_srcs):
        """Open a separate dialog with 2-D scatter and lasso selector."""
        # Determine indices to include
        mask = np.ones(self.viewer._coords_all.shape[0], dtype=bool)
        if selected_srcs and "source" in self.viewer._adata.obs:
            mask = self.viewer._adata.obs["source"].astype(str).isin(selected_srcs).values

        coords2d = self.viewer._coords_all[mask][:, :2]  # drop z
        indices = np.where(mask)[0]

        # Cluster labels for color coding
        clusters = None
        if "clusters" in self.viewer._adata.obs:
            clusters = self.viewer._adata.obs.iloc[indices]["clusters"].astype(str).to_numpy()

        if coords2d.shape[0] == 0:
            QMessageBox.information(self.viewer, "No Cells", "No cells in the selected section(s).")
            return

        dialog = LassoSelectionDialog(coords2d, indices, clusters, self.viewer)
        dialog.exec_()

    def start_circle_selection(self):
        """Start interactive circle selection for spatial analysis."""
        if self.viewer._adata is None or self.viewer._coords_all is None:
            QMessageBox.warning(self.viewer, "No Data", "Load data and render first.")
            return

        # Disable any previous picking to avoid PyVista error
        try:
            self.viewer._plotter_widget.disable_picking()
        except Exception:
            pass

        def _picked(picked_point):
            if picked_point is None or len(picked_point) == 0:
                return
            center = np.array(picked_point)
            # Ask for radius
            rad, ok = QInputDialog.getDouble(
                self.viewer, 
                "Circle Radius", 
                "Enter radius (same units as coords):", 
                50.0, 0.1, 1e6, 1
            )
            if not ok:
                return
            
            # Query KD-tree or direct distance
            if self.viewer._kd_tree is not None:
                idxs = self.viewer._kd_tree.query_ball_point(center, r=rad)
            else:
                dists = np.linalg.norm(self.viewer._coords_all - center, axis=1)
                idxs = np.where(dists <= rad)[0].tolist()
            
            if not idxs:
                QMessageBox.information(self.viewer, "Selection", "No cells within radius.")
                return
            
            self.process_selection(self.viewer._coords_all[idxs])

        # Enable single point picking
        self.viewer._plotter_widget.enable_point_picking(
            callback=lambda mesh: _picked(mesh.points[0]), 
            show_message=True
        )

    def start_point_selection(self):
        """Start interactive point selection for individual cell analysis."""
        if self.viewer._adata is None or self.viewer._coords_all is None:
            QMessageBox.warning(self.viewer, "No Data", "Load data and render first.")
            return

        try:
            self.viewer._plotter_widget.disable_picking()
        except Exception:
            pass

        def _picked(mesh):
            if mesh is None or mesh.n_points == 0:
                return
            pt = mesh.points[0]
            self.process_selection(np.array([pt]))

        self.viewer._plotter_widget.enable_point_picking(
            callback=_picked, 
            show_message=True
        )

    def process_selection(self, selected_pts: np.ndarray):
        """Given a set of selected XYZ coordinates, compute stats and plot.
        
        Args:
            selected_pts: Array of selected 3D coordinates (n_points, 3)
        """
        if self.viewer._coords_all is None:
            return

        # Match selected coordinates back to indices – use tolerance for float comparisons
        sel_indices = []
        for pt in selected_pts:
            # Compare with tolerance 1e-5
            dists = np.linalg.norm(self.viewer._coords_all - pt, axis=1)
            idx = np.where(dists < 1e-5)[0]
            if idx.size:
                sel_indices.extend(idx.tolist())
        
        if not sel_indices:
            QMessageBox.information(self.viewer, "Selection", "No cells matched the selected region.")
            return

        sel_indices = np.unique(sel_indices)

        if "clusters" not in self.viewer._adata.obs:
            QMessageBox.information(self.viewer, "Missing Clusters", "AnnData lacks 'clusters' information for analysis.")
            return
        
        # Perform composition analysis
        results = self.compute_composition_analysis(sel_indices)
        
        # Display results
        self.display_composition_results(results)
        
        # Highlight selected region in viewer
        self.highlight_selection(selected_pts)

    def compute_composition_analysis(self, sel_indices: np.ndarray):
        """Compute cell type composition analysis for selected cells.
        
        Args:
            sel_indices: Array of selected cell indices
            
        Returns:
            dict: Analysis results containing counts, percentages, and metadata
        """
        clusters = self.viewer._adata.obs.iloc[sel_indices]["clusters"].astype(str)
        counts = clusters.value_counts()
        percentages = counts / counts.sum() * 100
        
        # Additional statistics
        total_cells = len(sel_indices)
        unique_types = len(counts)
        
        # Get additional metadata if available
        metadata = {}
        if "source" in self.viewer._adata.obs:
            sources = self.viewer._adata.obs.iloc[sel_indices]["source"].astype(str)
            metadata["sources"] = sources.value_counts()
        
        if "n_counts" in self.viewer._adata.obs:
            n_counts = self.viewer._adata.obs.iloc[sel_indices]["n_counts"]
            metadata["mean_counts"] = n_counts.mean()
            metadata["median_counts"] = n_counts.median()
        
        return {
            "counts": counts,
            "percentages": percentages,
            "total_cells": total_cells,
            "unique_types": unique_types,
            "metadata": metadata,
            "selected_indices": sel_indices
        }

    def display_composition_results(self, results: dict):
        """Display composition analysis results in a dialog.
        
        Args:
            results: Analysis results from compute_composition_analysis
        """
        # Create dialog for results
        dlg = CompositionAnalysisDialog(results, self.viewer)
        dlg.exec_()

    def highlight_selection(self, selected_pts: np.ndarray):
        """Highlight selected region in the 3D viewer.
        
        Args:
            selected_pts: Array of selected 3D coordinates
        """
        # Remove previous selection highlight
        if self._analysis_selection_actor is not None:
            try:
                self.viewer._plotter_widget.remove_actor(self._analysis_selection_actor)
            except Exception:
                pass
        
        # Add new selection highlight
        self._analysis_selection_actor = self.viewer._plotter_widget.add_mesh(
            pv.PolyData(selected_pts),
            color="yellow",
            opacity=0.3,
            point_size=8,
            render_points_as_spheres=True,
        )
        self.viewer._plotter_widget.render()

    def export_selection_data(self, results: dict, filepath: str):
        """Export analysis results to file.
        
        Args:
            results: Analysis results dictionary
            filepath: Output file path
        """
        import pandas as pd
        
        # Create export dataframe
        export_data = {
            "cell_type": results["counts"].index,
            "count": results["counts"].values,
            "percentage": results["percentages"].values
        }
        
        df = pd.DataFrame(export_data)
        
        # Add metadata
        metadata_text = f"""
# Spatial Omics Analysis Results
# Total cells selected: {results['total_cells']}
# Unique cell types: {results['unique_types']}
# Analysis timestamp: {pd.Timestamp.now()}

"""
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(metadata_text)
        
        df.to_csv(filepath, mode='a', index=False)
        
        QMessageBox.information(
            self.viewer, 
            "Export Complete", 
            f"Analysis results exported to:\n{filepath}"
        )


class CompositionAnalysisDialog(QDialog):
    """Dialog for displaying detailed composition analysis results."""
    
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Type Composition Analysis")
        self.resize(600, 500)
        self.results = results
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the dialog UI with plots and statistics."""
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main bar plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_composition_bar(ax1)
        
        # Pie chart
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_composition_pie(ax2)
        
        # Statistics text
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_statistics_text(ax3)
        
        fig.tight_layout()
        
        # Add export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        from PyQt5.QtWidgets import QPushButton
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self._export_results)
        export_layout.addWidget(export_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        export_layout.addWidget(close_btn)
        
        layout.addLayout(export_layout)
    
    def _plot_composition_bar(self, ax):
        """Plot bar chart of cell type composition."""
        perc = self.results["percentages"]
        
        # Use colors that match the main viewer if possible
        bars = ax.bar(perc.index, perc.values, color="#1f77b4", alpha=0.7)
        
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"Cell Type Composition ({self.results['total_cells']} cells)")
        ax.set_xticklabels(perc.index, rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, value in zip(bars, perc.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    def _plot_composition_pie(self, ax):
        """Plot pie chart of cell type composition."""
        perc = self.results["percentages"]
        
        # Only show top 8 categories, group others
        if len(perc) > 8:
            top_cats = perc.head(7)
            other_sum = perc.tail(len(perc) - 7).sum()
            plot_data = top_cats.copy()
            plot_data["Others"] = other_sum
        else:
            plot_data = perc
        
        wedges, texts, autotexts = ax.pie(
            plot_data.values, 
            labels=plot_data.index,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title("Composition Overview")
        
        # Make percentage text smaller if many categories
        for autotext in autotexts:
            autotext.set_fontsize(8)
    
    def _plot_statistics_text(self, ax):
        """Display summary statistics as text."""
        ax.axis('off')
        
        stats_text = f"""Summary Statistics:

Total cells: {self.results['total_cells']:,}
Unique cell types: {self.results['unique_types']}

Most abundant: 
{self.results['percentages'].index[0]} 
({self.results['percentages'].iloc[0]:.1f}%)

Least abundant: 
{self.results['percentages'].index[-1]} 
({self.results['percentages'].iloc[-1]:.1f}%)
"""
        
        # Add metadata if available
        if self.results['metadata']:
            meta = self.results['metadata']
            if 'mean_counts' in meta:
                stats_text += f"\nMean UMI counts: {meta['mean_counts']:.0f}"
            if 'sources' in meta:
                n_sources = len(meta['sources'])
                stats_text += f"\nSections represented: {n_sources}"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def _export_results(self):
        """Export analysis results to CSV file."""
        from PyQt5.QtWidgets import QFileDialog
        import os
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Analysis Results", 
            "composition_analysis.csv", 
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if filename:
            # Use the analysis tools export function
            if hasattr(self.parent(), '_analysis_tools'):
                self.parent()._analysis_tools.export_selection_data(self.results, filename)
            else:
                # Fallback simple export
                import pandas as pd
                df = pd.DataFrame({
                    'cell_type': self.results['percentages'].index,
                    'percentage': self.results['percentages'].values,
                    'count': self.results['counts'].values
                })
                df.to_csv(filename, index=False)
                QMessageBox.information(self, "Export Complete", f"Results saved to {filename}")


def create_analysis_tools(viewer):
    """Factory function to create analysis tools instance.
    
    Args:
        viewer: Main SpatialOmicsViewer instance
        
    Returns:
        AnalysisTools: Configured analysis tools instance
    """
    return AnalysisTools(viewer) 