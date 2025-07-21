"""Analysis tools for spatial omics data exploration."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog
)
from scipy.spatial import cKDTree
import pyvista as pv
import vtk

# Handle both direct execution and package import
try:
    from .dialogs import SectionSelectDialog
except (ImportError, ValueError):
    from dialogs import SectionSelectDialog


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
        """Enable polygon selection directly on the main 3D view."""
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
        self._enable_main_screen_polygon_selection(selected_srcs)

    def _enable_main_screen_polygon_selection(self, selected_srcs):
        """Enable polygon selection mode directly on the main 3D viewer."""
        # Determine indices to include
        mask = np.ones(self.viewer._coords_all.shape[0], dtype=bool)
        if selected_srcs and "source" in self.viewer._adata.obs:
            mask = self.viewer._adata.obs["source"].astype(str).isin(selected_srcs).values

        if not mask.any():
            QMessageBox.information(self.viewer, "No Cells", "No cells in the selected section(s).")
            return

        # Store selection info
        self._polygon_selection_mask = mask
        self._polygon_selection_sources = selected_srcs
        self._polygon_points = []
        self._polygon_actors = []

        # Activate 2-D flattening so Z is ignored while selecting
        self._flatten_z = True
        
        # Store original plotter state to restore later
        self._original_camera_position = self.viewer._plotter_widget.camera_position
        
        # Clear and re-render only selected sections
        self._render_selected_sections_only()
        
        # Add "Selection Mode Enabled" text overlay
        self._add_selection_mode_overlay()
        
        # Lock camera to 2D view (looking down at x-y plane)
        self._lock_to_2d_view()
        
        # Enable polygon point picking
        self._enable_polygon_picking()
        
        # Add instructions
        QMessageBox.information(
            self.viewer, 
            "Polygon Selection Active", 
            "🔺 Polygon Selection Mode Enabled!\n\n"
            "• Click to add polygon vertices\n"
            "• Shift+click and drag to pan around\n"
            "• Right-click or press ENTER to finish polygon\n"
            "• ESC to exit selection mode\n"
            "• View is locked to X-Y coordinates\n"
            "• Only selected sections are visible"
        )

    def _render_selected_sections_only(self):
        """Re-render the view to show only selected sections with exact same colors as main screen."""
        # Clear the current view
        self.viewer._plotter_widget.clear()
        
        # Get coordinates for selected sections only
        coords_selected = self.viewer._coords_all[self._polygon_selection_mask]

        # If flattening requested, zero-out Z so the scene is purely 2-D
        if getattr(self, "_flatten_z", False):
            coords_selected = coords_selected.copy()
            coords_selected[:, 2] = 0  # collapse onto X-Y plane
        
        if coords_selected.shape[0] == 0:
            return
        
        # Use the EXACT same color logic as the main screen's _render_spatial method
        if "clusters" in self.viewer._adata.obs and coords_selected.shape[0] > 0:
            clusters_series = self.viewer._adata.obs["clusters"].astype(str)
            clusters_selected = clusters_series[self._polygon_selection_mask]

            colours = None  # will populate below

            # Generate colors using EXACT same logic as main screen _render_spatial
            all_cats = np.sort(np.unique(clusters_series))
            
            if self.viewer._color_scheme == "anndata" and "clusters_colors" in self.viewer._adata.obs:
                try:
                    # Get colors for selected cells directly from obs
                    colors_selected = self.viewer._adata.obs["clusters_colors"][self._polygon_selection_mask]
                    # Convert color strings to RGB values
                    import matplotlib.colors as mcolors
                    colours = np.vstack([mcolors.to_rgb(str(c)) for c in colors_selected]) * 255
                except Exception:
                    colours = None  # fallback if color parsing fails
            
            # Apply selected color scheme or fallback (same as main screen)
            if colours is None:
                try:
                    if self.viewer._color_scheme == "plotly_d3":
                        from .colors import generate_plotly_extended_palette
                        palette = generate_plotly_extended_palette(len(all_cats))
                    elif self.viewer._color_scheme == "custom_turbo":
                        from .colors import generate_custom_turbo_palette
                        palette = generate_custom_turbo_palette(len(all_cats))
                    elif self.viewer._color_scheme == "sns_palette":
                        from .colors import generate_sns_palette
                        palette = generate_sns_palette(len(all_cats))
                    elif self.viewer._color_scheme == "milliomics":
                        from .colors import generate_milliomics_palette
                        palette = generate_milliomics_palette(len(all_cats))
                    else:  # fallback to plotly_d3
                        from .colors import generate_plotly_extended_palette
                        palette = generate_plotly_extended_palette(len(all_cats))
                except (ImportError, ValueError):
                    # Fall back to absolute imports
                    if self.viewer._color_scheme == "plotly_d3":
                        from colors import generate_plotly_extended_palette
                        palette = generate_plotly_extended_palette(len(all_cats))
                    elif self.viewer._color_scheme == "custom_turbo":
                        from colors import generate_custom_turbo_palette
                        palette = generate_custom_turbo_palette(len(all_cats))
                    elif self.viewer._color_scheme == "sns_palette":
                        from colors import generate_sns_palette
                        palette = generate_sns_palette(len(all_cats))
                    elif self.viewer._color_scheme == "milliomics":
                        from colors import generate_milliomics_palette
                        palette = generate_milliomics_palette(len(all_cats))
                    else:  # fallback to plotly_d3
                        from colors import generate_plotly_extended_palette
                        palette = generate_plotly_extended_palette(len(all_cats))
                
                lookup = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}
                colours = np.vstack([lookup[c] for c in clusters_selected]) * 255
        else:
            colours = np.full((len(coords_selected), 3), 255, dtype=np.uint8)
        
        # Add selected sections with colors using same parameters as main screen
        cloud_selected = pv.PolyData(coords_selected)
        cloud_selected["colors"] = colours
        self.viewer._plotter_widget.add_mesh(
            cloud_selected,
            scalars="colors",
            rgb=True,
            point_size=5,  # Same as main screen
            render_points_as_spheres=True,
            opacity=self.viewer._point_opacity,  # Same opacity as main screen
        )
        
        # Add invisible cloud for picking (only selected sections)
        self.viewer._plotter_widget.add_mesh(
            pv.PolyData(coords_selected),
            color=(1, 1, 1),
            opacity=0.0,
            point_size=10,
            pickable=True,
            name="_polygon_pick_cloud",
            render_points_as_spheres=False,
        )
        
        # Set background
        self.viewer._plotter_widget.set_background("black")
        self.viewer._plotter_widget.render()

    def _add_selection_mode_overlay(self):
        """Add 'Selection Mode Enabled' text overlay to top-left corner."""
        try:
            # Remove any existing selection overlay
            if hasattr(self, '_selection_text_actor'):
                self.viewer._plotter_widget.remove_actor(self._selection_text_actor)
            
            # Add new text overlay with Arial font
            self._selection_text_actor = self.viewer._plotter_widget.add_text(
                "🔺 Selection Mode Enabled\nClick to add polygon points • Shift+click to pan • Right-click to finish • ESC to exit",
                position="upper_left",
                font_size=13,
                color="yellow",
                font="arial",
                name="selection_overlay"
            )
        except Exception as e:
            print(f"Could not add text overlay: {e}")

    def _lock_to_2d_view(self):
        """Lock camera to look down at x-y plane (disable z rotation)."""
        try:
            # Calculate bounds for the 2D view using only selected sections
            coords = self.viewer._coords_all[self._polygon_selection_mask]
            x_center = (coords[:, 0].max() + coords[:, 0].min()) / 2
            y_center = (coords[:, 1].max() + coords[:, 1].min()) / 2
            z_center = (coords[:, 2].max() + coords[:, 2].min()) / 2
            
            # Set camera to look down at x-y plane
            view_distance = max(coords[:, 0].ptp(), coords[:, 1].ptp()) * 1.2
            camera_pos = [x_center, y_center, z_center + view_distance]
            focal_point = [x_center, y_center, z_center]
            view_up = [0, 1, 0]  # Y-axis up
            
            # Set camera position
            self.viewer._plotter_widget.camera_position = [camera_pos, focal_point, view_up]
            self.viewer._plotter_widget.camera.parallel_projection = True
            
            # Store original settings before making changes
            interactor = self.viewer._plotter_widget.interactor
            camera = self.viewer._plotter_widget.camera
            
            # Store original interactor style and camera settings
            self._original_interactor_style = interactor.GetInteractorStyle()
            self._original_camera_settings = {
                'position': list(camera.position),
                'focal_point': list(camera.focal_point),
                'view_up': list(camera.view_up),
                'parallel_projection': camera.parallel_projection
            }
            
            # Create a limited interactor style for 2-D interaction only
            # vtkInteractorStyleImage supports pan/zoom but not rotation
            style = vtk.vtkInteractorStyleImage()
            interactor.SetInteractorStyle(style)
            
            print("Successfully locked to 2D view")  # Debug info
            self.viewer._plotter_widget.render()
            
        except Exception as e:
            print(f"Could not lock to 2D view: {e}")
            # Continue anyway - the selection will still work, just won't be locked to 2D

    def _enable_polygon_picking(self):
        """Enable point picking for polygon creation."""
        try:
            # Disable any existing picking
            self.viewer._plotter_widget.disable_picking()
            
            # Initialize panning state
            self._is_panning = False
            self._pan_start_pos = None
            self._pan_last_pos = None
            
            # Add mouse event observers
            iren = self.viewer._plotter_widget.interactor
            # Capture observer tags so we can remove them cleanly later
            self._polygon_observer_tags = [
                iren.AddObserver("LeftButtonPressEvent", self._on_polygon_mouse_press),
                iren.AddObserver("LeftButtonReleaseEvent", self._on_polygon_mouse_release),
                iren.AddObserver("MouseMoveEvent", self._on_polygon_mouse_move),
                iren.AddObserver("RightButtonPressEvent", self._finish_polygon_selection),
                iren.AddObserver("KeyPressEvent", self._on_key_press_polygon),
            ]

            # Block camera rotation events to keep view locked in 2-D.
            def _block_event(obj, evt):
                # Abort the event so default style won’t act on it
                iren.SetAbortFlag(1)
                iren.InvokeEvent("AbortCheckEvent")

            for _evt in ("StartRotateEvent", "RotateEvent", "EndRotateEvent",
                          "StartSpinEvent", "SpinEvent", "EndSpinEvent"):
                self._polygon_observer_tags.append(
                    iren.AddObserver(_evt, _block_event, 1.0)  # high priority ⇒ handled first
                )
            
        except Exception as e:
            print(f"Could not enable polygon picking: {e}")

    def _on_polygon_mouse_press(self, obj, event):
        """Handle left mouse press - check for shift+click for panning."""
        try:
            # Check if shift is pressed
            interactor = self.viewer._plotter_widget.interactor
            shift_pressed = interactor.GetShiftKey()
            
            if shift_pressed:
                # Start panning mode
                self._is_panning = True
                x, y = interactor.GetEventPosition()
                self._pan_start_pos = np.array([x, y])
                self._pan_last_pos = np.array([x, y])
                # Store initial camera position
                self._pan_start_camera = self.viewer._plotter_widget.camera_position

                # Prevent default interactor style from processing this press (avoids duplicate pan)
                interactor.SetAbortFlag(1)
                interactor.InvokeEvent("AbortCheckEvent")
            else:
                # Regular polygon point selection
                self._add_polygon_point_from_click()
                
        except Exception as e:
            print(f"Error in polygon mouse press: {e}")

    def _on_polygon_mouse_release(self, obj, event):
        """Handle left mouse release - end panning if active."""
        self._is_panning = False
        self._pan_start_pos = None
        self._pan_last_pos = None

    def _on_polygon_mouse_move(self, obj, event):
        """Handle mouse movement for panning."""
        interactor = self.viewer._plotter_widget.interactor

        shift_pressed = interactor.GetShiftKey()
        left_pressed = interactor.GetLeftButton()

        # If shift is held but mouse button not pressed, block the event to avoid unintended movement
        if shift_pressed and not left_pressed:
            interactor.SetAbortFlag(1)
            interactor.InvokeEvent("AbortCheckEvent")
            return

        # Only pan if we're actively panning AND shift is still being held
        if not self._is_panning or self._pan_start_pos is None or not shift_pressed:
            return
            
        try:
            # Get current mouse position
            x, y = self.viewer._plotter_widget.interactor.GetEventPosition()
            current_pos = np.array([x, y])
            
            # Calculate movement delta
            delta = current_pos - self._pan_last_pos
            self._pan_last_pos = current_pos
            
            # Apply panning - convert screen space movement to world space
            camera = self.viewer._plotter_widget.camera
            
            # Get current camera parameters
            position = np.array(camera.position)
            focal_point = np.array(camera.focal_point)
            
            # Calculate pan factors based on view distance and screen size
            view_distance = np.linalg.norm(position - focal_point)
            renderer = self.viewer._plotter_widget.renderer
            size = renderer.GetSize()
            pan_factor = view_distance / max(size)
            
            # Apply movement (invert Y for natural panning feel)
            dx = -delta[0] * pan_factor
            dy = delta[1] * pan_factor
            
            # Update camera position and focal point
            new_position = position + np.array([dx, dy, 0])
            new_focal_point = focal_point + np.array([dx, dy, 0])
            
            # Set new camera position
            camera.position = new_position
            camera.focal_point = new_focal_point
            
            self.viewer._plotter_widget.render()

            # Block default processing of this move event as we've handled camera update
            interactor.SetAbortFlag(1)
            interactor.InvokeEvent("AbortCheckEvent")

        except Exception as e:
            print(f"Error in panning: {e}")

    def _add_polygon_point_from_click(self):
        """Add polygon point from current click position."""
        try:
            # Get click position
            x, y = self.viewer._plotter_widget.interactor.GetEventPosition()
            
            # Pick the 3D point at this screen position
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(x, y, 0, self.viewer._plotter_widget.renderer)
            world_pos = picker.GetPickPosition()
            
            # Convert to numpy array
            point = np.array(world_pos)
            
            # Find closest actual data point to snap to
            coords_selected = self.viewer._coords_all[self._polygon_selection_mask]
            if len(coords_selected) == 0:
                return
                
            # Find closest point in selected data
            distances = np.linalg.norm(coords_selected - point, axis=1)
            closest_idx = np.argmin(distances)
            snapped_point = coords_selected[closest_idx].copy()
            if getattr(self, "_flatten_z", False):
                snapped_point[2] = 0  # keep on XY plane
            
            # Add this point to polygon
            self._add_polygon_point(snapped_point)
            
        except Exception as e:
            print(f"Error in adding polygon point: {e}")

    def _add_polygon_point(self, point):
        """Add a polygon vertex and update visualization."""
        if getattr(self, "_flatten_z", False):
            point = point.copy(); point[2] = 0

        self._polygon_points.append(point)
        
        # Calculate smaller adaptive radius based on selected data scale
        coords = self.viewer._coords_all[self._polygon_selection_mask]
        scene_scale = max(coords[:, 0].ptp(), coords[:, 1].ptp(), coords[:, 2].ptp())
        sphere_radius = scene_scale * 0.003  # Reduced from 0.01 to 0.003 for smaller points
        
        # Add visual marker for the point
        sphere = pv.Sphere(radius=sphere_radius, center=point)
        actor = self.viewer._plotter_widget.add_mesh(
            sphere,
            color="yellow",
            opacity=0.9,
            name=f"polygon_point_{len(self._polygon_points)}"
        )
        self._polygon_actors.append(actor)
        
        # If we have more than one point, draw line between last two points
        if len(self._polygon_points) > 1:
            # Create line between last two points
            p_prev = self._polygon_points[-2].copy()
            p_curr = self._polygon_points[-1].copy()
            if getattr(self, "_flatten_z", False):
                p_prev[2] = p_curr[2] = 0
            points = np.array([p_prev, p_curr])
            line = pv.Line(points[0], points[1])
            line_actor = self.viewer._plotter_widget.add_mesh(
                line,
                color="yellow",
                line_width=5,
                opacity=0.9,
                name=f"polygon_line_{len(self._polygon_points)}"
            )
            self._polygon_actors.append(line_actor)
        
        self.viewer._plotter_widget.render()
        
        # Update status
        self._update_selection_overlay(f"🔺 Selection Mode: {len(self._polygon_points)} points added")

    def _on_key_press_polygon(self, obj, event):
        """Handle key press events during polygon selection."""
        key = self.viewer._plotter_widget.interactor.GetKeySym()
        
        if key.lower() == 'escape':
            self._exit_polygon_selection()
        elif key.lower() in ['return', 'enter']:
            self._finish_polygon_selection()

    def _finish_polygon_selection(self, obj=None, event=None):
        """Finish polygon selection and process the selection."""
        if len(self._polygon_points) < 3:
            QMessageBox.warning(
                self.viewer, 
                "Insufficient Points", 
                "Need at least 3 points to create a polygon."
            )
            return
        
        # Close the polygon by connecting last point to first
        if len(self._polygon_points) > 2:
            points = np.array([self._polygon_points[-1], self._polygon_points[0]])
            line = pv.Line(points[0], points[1])
            line_actor = self.viewer._plotter_widget.add_mesh(
                line,
                color="yellow",
                line_width=5,
                name="polygon_closing_line"
            )
            self._polygon_actors.append(line_actor)
            self.viewer._plotter_widget.render()
        
        # Process the polygon selection (this will show the 3-button confirmation)
        self._process_polygon_selection()

    def _process_polygon_selection(self):
        """Process the polygon selection to find cells inside."""
        # Convert 3D points to 2D for polygon checking
        polygon_2d = np.array([[p[0], p[1]] for p in self._polygon_points])
        path = mpath.Path(polygon_2d)
        
        # Get 2D coordinates of all cells in selected sections
        coords_2d = self.viewer._coords_all[self._polygon_selection_mask][:, :2]
        indices = np.where(self._polygon_selection_mask)[0]
        
        # Check which points are inside the polygon
        inside_mask = path.contains_points(coords_2d)
        
        if not inside_mask.any():
            QMessageBox.information(self.viewer, "Selection", "No cells inside polygon.")
            return
        
        # Get the selected indices
        selected_indices = indices[inside_mask]
        selected_coords = self.viewer._coords_all[selected_indices]
        
        # Highlight the selected cells before processing
        self._highlight_selected_cells(selected_coords)
        
        # Create custom confirmation dialog with three buttons
        msg_box = QMessageBox(self.viewer)
        msg_box.setWindowTitle("Confirm Selection")
        msg_box.setText(f"Found {len(selected_indices)} cells inside polygon.")
        msg_box.setInformativeText("What would you like to do?")
        msg_box.setIcon(QMessageBox.Question)
        
        # Add custom buttons in order from left to right
        quit_btn = msg_box.addButton("Quit Selection", QMessageBox.RejectRole)
        reselect_btn = msg_box.addButton("Reselect", QMessageBox.ResetRole) 
        continue_btn = msg_box.addButton("Continue", QMessageBox.AcceptRole)
        
        # Set the default button (Continue) and make it prominent
        msg_box.setDefaultButton(continue_btn)
        
        # Style the dialog for better appearance
        button_style = """
            QPushButton {
                background-color: #404040;
                color: white;
                border: 2px solid #606060;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #707070;
            }
            QPushButton:pressed {
                background-color: #353535;
                border-color: #505050;
            }
        """
        
        continue_style = button_style + """
            QPushButton {
                background-color: #2196F3;
                border-color: #1976D2;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #1565C0;
                border-color: #0D47A1;
            }
        """
        
        # Apply styles to buttons
        quit_btn.setStyleSheet(button_style)
        reselect_btn.setStyleSheet(button_style)
        continue_btn.setStyleSheet(continue_style)
        
        msg_box.setStyleSheet("""
            QMessageBox {
                font-size: 12px;
                background-color: #2b2b2b;
                color: white;
            }
        """)
        
        # Execute the dialog
        msg_box.exec_()
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == continue_btn:
            # Process the selection and exit
            self._remove_selection_highlight()
            self.process_selection(selected_coords)
            self._exit_polygon_selection()
        elif clicked_button == reselect_btn:
            # Clear current polygon and start over
            self._remove_selection_highlight()
            self._clear_current_polygon()
            self._update_selection_overlay("🔺 Selection Mode: Reselecting - click to add new polygon points")
        else:  # quit_btn
            # Exit selection mode completely
            self._remove_selection_highlight()
            self._exit_polygon_selection()

    def _clear_current_polygon(self):
        """Clear the current polygon points and visual elements."""
        # Remove all current polygon visual elements
        for actor in self._polygon_actors:
            try:
                self.viewer._plotter_widget.remove_actor(actor)
            except:
                pass
        
        # Clear polygon data
        self._polygon_points = []
        self._polygon_actors = []
        
        self.viewer._plotter_widget.render()

    def _highlight_selected_cells(self, selected_coords):
        """Temporarily highlight selected cells for confirmation."""
        try:
            # Respect 2-D flattening during selection
            if getattr(self, "_flatten_z", False):
                selected_coords = selected_coords.copy()
                selected_coords[:, 2] = 0

            # Calculate radius for highlighting
            coords = self.viewer._coords_all[self._polygon_selection_mask]
            scene_scale = max(coords[:, 0].ptp(), coords[:, 1].ptp(), coords[:, 2].ptp())
            highlight_radius = scene_scale * 0.005
            
            # Create highlight mesh
            points = pv.PolyData(selected_coords)
            self._selection_highlight_actor = self.viewer._plotter_widget.add_mesh(
                points,
                color="red",
                point_size=8,
                opacity=0.8,
                render_points_as_spheres=True,
                name="selection_highlight"
            )
            self.viewer._plotter_widget.render()
        except Exception as e:
            print(f"Could not highlight selection: {e}")

    def _remove_selection_highlight(self):
        """Remove selection highlight."""
        try:
            if hasattr(self, '_selection_highlight_actor') and self._selection_highlight_actor is not None:
                self.viewer._plotter_widget.remove_actor(self._selection_highlight_actor)
                self._selection_highlight_actor = None
                self.viewer._plotter_widget.render()
        except Exception as e:
            print(f"Could not remove highlight: {e}")

    def _exit_polygon_selection(self):
        """Exit polygon selection mode and restore normal view."""
        # Remove all polygon visual elements
        for actor in self._polygon_actors:
            try:
                self.viewer._plotter_widget.remove_actor(actor)
            except:
                pass
        
        # Remove selection overlay text
        if hasattr(self, '_selection_text_actor'):
            try:
                self.viewer._plotter_widget.remove_actor(self._selection_text_actor)
            except:
                pass
        
        # Remove any selection highlights
        self._remove_selection_highlight()
        
        # Restore original camera settings
        if hasattr(self, '_original_camera_settings'):
            try:
                camera = self.viewer._plotter_widget.camera
                settings = self._original_camera_settings
                camera.position = settings['position']
                camera.focal_point = settings['focal_point']
                camera.view_up = settings['view_up']
                camera.parallel_projection = settings['parallel_projection']
            except Exception as e:
                print(f"Could not restore camera settings: {e}")
        
        # Restore original camera position if available
        if hasattr(self, '_original_camera_position'):
            try:
                self.viewer._plotter_widget.camera_position = self._original_camera_position
            except Exception as e:
                print(f"Could not restore camera position: {e}")
        
        # Restore original interactor style
        if hasattr(self, '_original_interactor_style'):
            try:
                self.viewer._plotter_widget.interactor.SetInteractorStyle(self._original_interactor_style)
            except Exception as e:
                print(f"Could not restore interactor style: {e}")
        else:
            # Fallback: set default trackball camera style
            try:
                default_style = vtk.vtkInteractorStyleTrackballCamera()
                self.viewer._plotter_widget.interactor.SetInteractorStyle(default_style)
            except Exception as e:
                print(f"Could not set default interactor style: {e}")
        
        # Disable picking
        try:
            self.viewer._plotter_widget.disable_picking()
        except:
            pass
        
        # Remove observers
        # Clean up only the observers we added so default interaction remains intact
        if hasattr(self, '_polygon_observer_tags'):
            iren = self.viewer._plotter_widget.interactor
            for tag in self._polygon_observer_tags:
                try:
                    iren.RemoveObserver(tag)
                except Exception as e:
                    print(f"Could not remove observer {tag}: {e}")
            del self._polygon_observer_tags
        
        # Restore original data view (re-render all data as it was)
        try:
            if self.viewer._mode == "cell":
                self.viewer._render_spatial()
            elif self.viewer._mode == "gene":
                self.viewer._render_gene_mode()
            elif self.viewer._mode == "gene_spots":
                self.viewer._render_gene_only()
        except:
            pass
        
        # Clear stored data
        self._polygon_points = []
        self._polygon_actors = []
        if hasattr(self, '_polygon_selection_mask'):
            del self._polygon_selection_mask
        if hasattr(self, '_polygon_selection_sources'):
            del self._polygon_selection_sources
        if hasattr(self, '_original_camera_settings'):
            del self._original_camera_settings
        if hasattr(self, '_flatten_z'):
            del self._flatten_z
        
        self.viewer._plotter_widget.render()
        
        QMessageBox.information(self.viewer, "Selection Complete", "Polygon selection mode exited. View restored.")

    def _update_selection_overlay(self, text):
        """Update the selection mode overlay text."""
        try:
            if hasattr(self, '_selection_text_actor'):
                self.viewer._plotter_widget.remove_actor(self._selection_text_actor)
            
            self._selection_text_actor = self.viewer._plotter_widget.add_text(
                text + "\nShift+click to pan • Right-click to finish • ESC to exit",
                position="upper_left",
                font_size=13,
                color="yellow",
                font="arial",
                name="selection_overlay"
            )
        except Exception as e:
            print(f"Could not update overlay: {e}")

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
        
        # Create matplotlib figure with better size and DPI
        fig = Figure(figsize=(10, 7), dpi=100)
        fig.patch.set_facecolor('white')
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Add interactive toolbar for zooming/panning
        try:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setStyleSheet("QToolBar { background: #2b2b2b; border: 1px solid #555; }")
            layout.addWidget(toolbar)
        except Exception:
            # Fallback: if toolbar import fails, continue without it
            pass

        # Create subplots with better spacing
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4, 
                             left=0.1, right=0.95, top=0.93, bottom=0.15)
        
        # Main bar plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_composition_bar(ax1)
        
        # Pie chart
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_composition_pie(ax2)
        
        # Statistics text
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_statistics_text(ax3)
        
        # Use tight_layout with padding
        fig.tight_layout(pad=2.0)
        
        # Add export button with main screen styling
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        # Apply main screen button style
        button_style = """
            QPushButton {
                background-color: #404040;
                color: white;
                border: 2px solid #606060;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #707070;
            }
            QPushButton:pressed {
                background-color: #353535;
                border-color: #505050;
            }
        """
        
        from PyQt5.QtWidgets import QPushButton
        export_btn = QPushButton("Export Results")
        export_btn.setStyleSheet(button_style)
        export_btn.clicked.connect(self._export_results)
        export_layout.addWidget(export_btn)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(button_style)
        close_btn.clicked.connect(self.accept)
        export_layout.addWidget(close_btn)
        
        layout.addLayout(export_layout)
    
    def _plot_composition_bar(self, ax):
        """Plot bar chart of cell type composition with improved text handling."""
        perc = self.results["percentages"]
        
        # Use colors that match the main viewer if possible
        bars = ax.bar(perc.index, perc.values, color="#1f77b4", alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel("Percentage (%)", fontsize=11, fontweight='bold')
        ax.set_title(f"Cell Type Composition ({self.results['total_cells']} cells)", 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Improve x-axis label handling to prevent overlap
        if len(perc.index) > 10:
            # For many categories, use smaller font and vertical rotation
            ax.set_xticklabels(perc.index, rotation=90, ha="center", fontsize=9)
        elif len(perc.index) > 5:
            # For moderate categories, use 45-degree rotation
            ax.set_xticklabels(perc.index, rotation=45, ha="right", fontsize=10)
        else:
            # For few categories, keep horizontal
            ax.set_xticklabels(perc.index, rotation=0, ha="center", fontsize=11)
        
        # Add value labels on bars with better positioning
        for bar, value in zip(bars, perc.values):
            height = bar.get_height()
            # Position text above bar with some padding
            label_y = height + max(perc.values) * 0.01
            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set y-axis limit to accommodate labels
        ax.set_ylim(0, max(perc.values) * 1.15)
        
        # Improve grid and styling
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    def _plot_composition_pie(self, ax):
        """Plot pie chart of cell type composition with better label handling."""
        perc = self.results["percentages"]
        
        # Only show top 8 categories, group others
        if len(perc) > 8:
            top_cats = perc.head(7)
            other_sum = perc.tail(len(perc) - 7).sum()
            plot_data = top_cats.copy()
            plot_data["Others"] = other_sum
        else:
            plot_data = perc
        
        # Use better color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        
        # Create pie chart with better label positioning
        wedges, texts, autotexts = ax.pie(
            plot_data.values, 
            labels=plot_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            pctdistance=0.85,
            labeldistance=1.1
        )
        
        ax.set_title("Composition Overview", fontsize=12, fontweight='bold', pad=15)
        
        # Improve text formatting
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        for text in texts:
            text.set_fontsize(9)
            # Truncate long labels
            if len(text.get_text()) > 12:
                text.set_text(text.get_text()[:10] + "...")
    
    def _plot_statistics_text(self, ax):
        """Display summary statistics as text with better formatting."""
        ax.axis('off')
        
        # Create more organized statistics text
        stats_lines = [
            "Summary Statistics:",
            "",
            f"Total cells: {self.results['total_cells']:,}",
            f"Unique cell types: {self.results['unique_types']}",
            "",
            "Most abundant:",
            f"{self.results['percentages'].index[0]}",
            f"({self.results['percentages'].iloc[0]:.1f}%)",
            "",
            "Least abundant:",
            f"{self.results['percentages'].index[-1]}",
            f"({self.results['percentages'].iloc[-1]:.1f}%)"
        ]
        
        # Add metadata if available
        if self.results['metadata']:
            meta = self.results['metadata']
            stats_lines.append("")
            if 'mean_counts' in meta:
                stats_lines.append(f"Mean UMI counts: {meta['mean_counts']:.0f}")
            if 'sources' in meta:
                n_sources = len(meta['sources'])
                stats_lines.append(f"Sections represented: {n_sources}")
        
        # Join lines and display with better formatting
        stats_text = "\n".join(stats_lines)
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
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