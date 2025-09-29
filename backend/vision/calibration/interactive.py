"""Interactive calibration tools."""

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from .camera import CameraCalibrator
from .color import ColorCalibrator
from .geometry import GeometricCalibrator
from .validation import CalibrationValidator

logger = logging.getLogger(__name__)


class InteractiveCalibrationGUI:
    """Interactive GUI for comprehensive calibration system."""

    def __init__(self, master: tk.Tk) -> None:
        """Initialize calibration GUI.

        Args:
            master: Root tkinter window
        """
        self.master = master
        self.master.title("Billiards Vision Calibration System")
        self.master.geometry("1200x800")

        # Initialize calibrators
        self.camera_calibrator = CameraCalibrator()
        self.color_calibrator = ColorCalibrator()
        self.geometry_calibrator = GeometricCalibrator()
        self.validator = CalibrationValidator()

        # GUI state
        self.current_frame = None
        self.calibration_images = []
        self.selected_corners = []
        self.color_samples = {}

        # Video capture
        self.cap = None
        self.capture_thread = None
        self.frame_queue = queue.Queue()
        self.is_capturing = False

        self.setup_gui()
        self.setup_opencv_windows()

    def setup_gui(self) -> None:
        """Setup the main GUI layout."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Camera calibration tab
        self.camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_frame, text="Camera Calibration")
        self.setup_camera_tab()

        # Color calibration tab
        self.color_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.color_frame, text="Color Calibration")
        self.setup_color_tab()

        # Geometry calibration tab
        self.geometry_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.geometry_frame, text="Geometry Calibration")
        self.setup_geometry_tab()

        # Validation tab
        self.validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.validation_frame, text="Validation")
        self.setup_validation_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.master, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_camera_tab(self) -> None:
        """Setup camera calibration tab."""
        # Camera controls
        camera_control_frame = ttk.LabelFrame(self.camera_frame, text="Camera Controls")
        camera_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            camera_control_frame, text="Start Camera", command=self.start_camera
        ).pack(side="left", padx=5)
        ttk.Button(
            camera_control_frame, text="Stop Camera", command=self.stop_camera
        ).pack(side="left", padx=5)
        ttk.Button(
            camera_control_frame,
            text="Capture Calibration Image",
            command=self.capture_calibration_image,
        ).pack(side="left", padx=5)

        # Calibration controls
        calibration_control_frame = ttk.LabelFrame(
            self.camera_frame, text="Calibration Controls"
        )
        calibration_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            calibration_control_frame,
            text="Calibrate Camera",
            command=self.calibrate_camera,
        ).pack(side="left", padx=5)
        ttk.Button(
            calibration_control_frame,
            text="Load Images",
            command=self.load_calibration_images,
        ).pack(side="left", padx=5)
        ttk.Button(
            calibration_control_frame,
            text="Clear Images",
            command=self.clear_calibration_images,
        ).pack(side="left", padx=5)

        # Status display
        status_frame = ttk.LabelFrame(self.camera_frame, text="Status")
        status_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.camera_status_text = tk.Text(status_frame, height=10)
        scrollbar = ttk.Scrollbar(
            status_frame, orient="vertical", command=self.camera_status_text.yview
        )
        self.camera_status_text.configure(yscrollcommand=scrollbar.set)

        self.camera_status_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_color_tab(self) -> None:
        """Setup color calibration tab."""
        # Color controls
        color_control_frame = ttk.LabelFrame(self.color_frame, text="Color Calibration")
        color_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            color_control_frame,
            text="Auto-Calibrate Table",
            command=self.auto_calibrate_table_color,
        ).pack(side="left", padx=5)
        ttk.Button(
            color_control_frame,
            text="Open Color Picker",
            command=self.open_color_picker,
        ).pack(side="left", padx=5)
        ttk.Button(
            color_control_frame,
            text="Save Color Profile",
            command=self.save_color_profile,
        ).pack(side="left", padx=5)
        ttk.Button(
            color_control_frame,
            text="Load Color Profile",
            command=self.load_color_profile,
        ).pack(side="left", padx=5)

        # Ball color selection
        ball_frame = ttk.LabelFrame(self.color_frame, text="Ball Color Calibration")
        ball_frame.pack(fill="x", padx=5, pady=5)

        self.ball_type_var = tk.StringVar(value="cue")
        ball_types = [
            "cue",
            "yellow",
            "blue",
            "red",
            "purple",
            "orange",
            "green",
            "brown",
            "black",
        ]

        ttk.Label(ball_frame, text="Ball Type:").pack(side="left", padx=5)
        ball_combo = ttk.Combobox(
            ball_frame,
            textvariable=self.ball_type_var,
            values=ball_types,
            state="readonly",
        )
        ball_combo.pack(side="left", padx=5)

        ttk.Button(
            ball_frame, text="Select Ball Regions", command=self.select_ball_regions
        ).pack(side="left", padx=5)

        # Color threshold adjustment
        threshold_frame = ttk.LabelFrame(self.color_frame, text="Threshold Adjustment")
        threshold_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # HSV sliders
        self.hsv_vars = {
            "hue_min": tk.IntVar(value=0),
            "hue_max": tk.IntVar(value=179),
            "sat_min": tk.IntVar(value=0),
            "sat_max": tk.IntVar(value=255),
            "val_min": tk.IntVar(value=0),
            "val_max": tk.IntVar(value=255),
        }

        row = 0
        for param, var in self.hsv_vars.items():
            ttk.Label(threshold_frame, text=param.replace("_", " ").title() + ":").grid(
                row=row, column=0, sticky="w", padx=5
            )
            max_val = 179 if "hue" in param else 255
            scale = ttk.Scale(
                threshold_frame,
                from_=0,
                to=max_val,
                orient="horizontal",
                variable=var,
                command=self.update_color_preview,
            )
            scale.grid(row=row, column=1, sticky="ew", padx=5)
            ttk.Label(threshold_frame, textvariable=var).grid(row=row, column=2, padx=5)
            row += 1

        threshold_frame.columnconfigure(1, weight=1)

    def setup_geometry_tab(self) -> None:
        """Setup geometry calibration tab."""
        # Geometry controls
        geometry_control_frame = ttk.LabelFrame(
            self.geometry_frame, text="Geometry Calibration"
        )
        geometry_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            geometry_control_frame,
            text="Auto-Detect Table",
            command=self.auto_detect_table,
        ).pack(side="left", padx=5)
        ttk.Button(
            geometry_control_frame,
            text="Manual Corner Selection",
            command=self.manual_corner_selection,
        ).pack(side="left", padx=5)
        ttk.Button(
            geometry_control_frame,
            text="Calibrate Geometry",
            command=self.calibrate_geometry,
        ).pack(side="left", padx=5)

        # Table dimensions
        dimensions_frame = ttk.LabelFrame(self.geometry_frame, text="Table Dimensions")
        dimensions_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(dimensions_frame, text="Width (m):").pack(side="left", padx=5)
        self.table_width_var = tk.DoubleVar(value=2.54)
        width_entry = ttk.Entry(
            dimensions_frame, textvariable=self.table_width_var, width=10
        )
        width_entry.pack(side="left", padx=5)

        ttk.Label(dimensions_frame, text="Height (m):").pack(side="left", padx=5)
        self.table_height_var = tk.DoubleVar(value=1.27)
        height_entry = ttk.Entry(
            dimensions_frame, textvariable=self.table_height_var, width=10
        )
        height_entry.pack(side="left", padx=5)

        # Corner coordinates display
        corners_frame = ttk.LabelFrame(self.geometry_frame, text="Table Corners")
        corners_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.corners_text = tk.Text(corners_frame, height=8)
        corners_scrollbar = ttk.Scrollbar(
            corners_frame, orient="vertical", command=self.corners_text.yview
        )
        self.corners_text.configure(yscrollcommand=corners_scrollbar.set)

        self.corners_text.pack(side="left", fill="both", expand=True)
        corners_scrollbar.pack(side="right", fill="y")

    def setup_validation_tab(self) -> None:
        """Setup validation tab."""
        # Validation controls
        validation_control_frame = ttk.LabelFrame(
            self.validation_frame, text="Validation Tests"
        )
        validation_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            validation_control_frame,
            text="Validate Camera",
            command=self.validate_camera,
        ).pack(side="left", padx=5)
        ttk.Button(
            validation_control_frame,
            text="Validate Colors",
            command=self.validate_colors,
        ).pack(side="left", padx=5)
        ttk.Button(
            validation_control_frame,
            text="Validate Geometry",
            command=self.validate_geometry,
        ).pack(side="left", padx=5)
        ttk.Button(
            validation_control_frame,
            text="Full Validation",
            command=self.full_validation,
        ).pack(side="left", padx=5)

        # Results display
        results_frame = ttk.LabelFrame(self.validation_frame, text="Validation Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.validation_results_text = tk.Text(results_frame)
        validation_scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.validation_results_text.yview
        )
        self.validation_results_text.configure(yscrollcommand=validation_scrollbar.set)

        self.validation_results_text.pack(side="left", fill="both", expand=True)
        validation_scrollbar.pack(side="right", fill="y")

    def setup_opencv_windows(self) -> None:
        """Setup OpenCV windows for interactive selection."""
        # Initialize OpenCV windows
        cv2.namedWindow("Calibration View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration View", 800, 600)

    def start_camera(self) -> None:
        """Start camera capture."""
        try:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return

            self.is_capturing = True
            self.capture_thread = threading.Thread(
                target=self.capture_loop, daemon=True
            )
            self.capture_thread.start()

            self.status_var.set("Camera started")
            self.update_camera_status("Camera capture started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self) -> None:
        """Stop camera capture."""
        self.is_capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        self.status_var.set("Camera stopped")
        self.update_camera_status("Camera capture stopped")

    def capture_loop(self) -> None:
        """Main camera capture loop."""
        while self.is_capturing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                # Display frame in OpenCV window
                display_frame = frame.copy()

                # Draw calibration overlay if needed
                if hasattr(self, "show_chessboard") and self.show_chessboard:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray, self.camera_calibrator.chessboard_size, None
                    )
                    if ret:
                        cv2.drawChessboardCorners(
                            display_frame,
                            self.camera_calibrator.chessboard_size,
                            corners,
                            ret,
                        )

                cv2.imshow("Calibration View", display_frame)
                cv2.waitKey(1)

    def capture_calibration_image(self) -> None:
        """Capture current frame for camera calibration."""
        if self.current_frame is not None:
            self.calibration_images.append(self.current_frame.copy())
            self.update_camera_status(
                f"Captured calibration image {len(self.calibration_images)}"
            )
            self.status_var.set(f"Calibration images: {len(self.calibration_images)}")

    def load_calibration_images(self) -> None:
        """Load calibration images from files."""
        file_paths = filedialog.askopenfilenames(
            title="Select Calibration Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )

        for path in file_paths:
            image = cv2.imread(path)
            if image is not None:
                self.calibration_images.append(image)

        self.update_camera_status(f"Loaded {len(file_paths)} calibration images")
        self.status_var.set(f"Calibration images: {len(self.calibration_images)}")

    def clear_calibration_images(self) -> None:
        """Clear all calibration images."""
        self.calibration_images.clear()
        self.update_camera_status("Cleared all calibration images")
        self.status_var.set("Calibration images: 0")

    def calibrate_camera(self) -> None:
        """Perform camera calibration."""
        if len(self.calibration_images) < 10:
            messagebox.showwarning("Warning", "Need at least 10 calibration images")
            return

        try:
            self.update_camera_status("Starting camera calibration...")
            self.status_var.set("Calibrating camera...")

            success, camera_params = self.camera_calibrator.calibrate_intrinsics(
                self.calibration_images
            )

            if success:
                self.update_camera_status("Camera calibration successful!")
                self.update_camera_status(
                    f"Calibration error: {camera_params.calibration_error:.3f} pixels"
                )
                self.update_camera_status(f"Resolution: {camera_params.resolution}")
                self.status_var.set("Camera calibration completed")
            else:
                self.update_camera_status("Camera calibration failed")
                messagebox.showerror("Error", "Camera calibration failed")

        except Exception as e:
            self.update_camera_status(f"Calibration error: {e}")
            messagebox.showerror("Error", f"Calibration failed: {e}")

    def auto_calibrate_table_color(self) -> None:
        """Auto-calibrate table color."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available for calibration")
            return

        try:
            thresholds = self.color_calibrator.auto_calibrate_table_color(
                self.current_frame
            )
            self.status_var.set("Table color auto-calibrated")

            # Update HSV sliders
            self.hsv_vars["hue_min"].set(thresholds.hue_min)
            self.hsv_vars["hue_max"].set(thresholds.hue_max)
            self.hsv_vars["sat_min"].set(thresholds.saturation_min)
            self.hsv_vars["sat_max"].set(thresholds.saturation_max)
            self.hsv_vars["val_min"].set(thresholds.value_min)
            self.hsv_vars["val_max"].set(thresholds.value_max)

            self.update_color_preview()

        except Exception as e:
            messagebox.showerror("Error", f"Auto-calibration failed: {e}")

    def open_color_picker(self) -> None:
        """Open interactive color picker."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return

        try:
            selected_colors = self.color_calibrator.create_color_picker_interface(
                self.current_frame, "Color Picker"
            )
            if selected_colors:
                self.status_var.set(f"Selected {len(selected_colors)} colors")
        except Exception as e:
            messagebox.showerror("Error", f"Color picker failed: {e}")

    def select_ball_regions(self) -> None:
        """Select regions for ball color calibration."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return

        ball_type = self.ball_type_var.get()
        self.status_var.set(
            f"Select regions for {ball_type} ball (press 'q' when done)"
        )

        # Use OpenCV's selectROI for region selection
        regions = []
        frame_copy = self.current_frame.copy()

        while True:
            roi = cv2.selectROI(f"Select {ball_type} regions", frame_copy, False)
            if roi[2] > 0 and roi[3] > 0:  # Valid ROI
                regions.append(roi)
                # Draw rectangle on frame
                cv2.rectangle(
                    frame_copy,
                    (roi[0], roi[1]),
                    (roi[0] + roi[2], roi[1] + roi[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(f"Select {ball_type} regions", frame_copy)
            else:
                break

        cv2.destroyWindow(f"Select {ball_type} regions")

        if regions:
            self.color_samples[ball_type] = regions
            self.status_var.set(f"Selected {len(regions)} regions for {ball_type}")

    def update_color_preview(self, *args) -> None:
        """Update color threshold preview."""
        if self.current_frame is None:
            return

        # Get current threshold values
        from .color import ColorThresholds

        thresholds = ColorThresholds(
            hue_min=self.hsv_vars["hue_min"].get(),
            hue_max=self.hsv_vars["hue_max"].get(),
            saturation_min=self.hsv_vars["sat_min"].get(),
            saturation_max=self.hsv_vars["sat_max"].get(),
            value_min=self.hsv_vars["val_min"].get(),
            value_max=self.hsv_vars["val_max"].get(),
        )

        # Apply mask
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        mask = thresholds.apply_mask(hsv)

        # Show preview
        cv2.imshow("Color Preview", mask)

    def save_color_profile(self) -> None:
        """Save current color profile."""
        if self.color_calibrator.current_profile is None:
            messagebox.showwarning("Warning", "No color profile to save")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Color Profile",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )

        if filename:
            success = self.color_calibrator.save_profile(
                self.color_calibrator.current_profile, Path(filename).name
            )
            if success:
                self.status_var.set("Color profile saved")
            else:
                messagebox.showerror("Error", "Failed to save color profile")

    def load_color_profile(self) -> None:
        """Load color profile."""
        filename = filedialog.askopenfilename(
            title="Load Color Profile", filetypes=[("JSON files", "*.json")]
        )

        if filename:
            profile = self.color_calibrator.load_profile(Path(filename).name)
            if profile:
                self.status_var.set("Color profile loaded")
            else:
                messagebox.showerror("Error", "Failed to load color profile")

    def auto_detect_table(self) -> None:
        """Auto-detect table corners."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return

        try:
            corners = self.geometry_calibrator.detect_table_corners(self.current_frame)
            self.selected_corners = corners
            self.update_corners_display()
            self.status_var.set("Table corners auto-detected")
        except Exception as e:
            messagebox.showerror("Error", f"Table detection failed: {e}")

    def manual_corner_selection(self) -> None:
        """Manual table corner selection."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return

        self.selected_corners = []
        self.status_var.set("Click on 4 table corners in clockwise order")

        # Setup mouse callback for corner selection
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.selected_corners) < 4:
                self.selected_corners.append((x, y))
                # Draw point
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    param,
                    str(len(self.selected_corners)),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Corner Selection", param)

                if len(self.selected_corners) == 4:
                    self.update_corners_display()
                    self.status_var.set("4 corners selected")

        frame_copy = self.current_frame.copy()
        cv2.namedWindow("Corner Selection")
        cv2.setMouseCallback("Corner Selection", mouse_callback, frame_copy)
        cv2.imshow("Corner Selection", frame_copy)

    def calibrate_geometry(self) -> None:
        """Perform geometric calibration."""
        if len(self.selected_corners) != 4:
            messagebox.showwarning("Warning", "Need 4 table corners")
            return

        try:
            table_dimensions = (self.table_width_var.get(), self.table_height_var.get())
            self.geometry_calibrator.calibrate_table_geometry(
                self.current_frame, self.selected_corners, table_dimensions
            )

            self.status_var.set("Geometric calibration completed")
            self.update_corners_display()

        except Exception as e:
            messagebox.showerror("Error", f"Geometric calibration failed: {e}")

    def update_corners_display(self) -> None:
        """Update corners display in text widget."""
        self.corners_text.delete(1.0, tk.END)
        if self.selected_corners:
            self.corners_text.insert(tk.END, "Selected Corners:\n")
            for i, (x, y) in enumerate(self.selected_corners):
                self.corners_text.insert(
                    tk.END, f"  Corner {i+1}: ({x:.1f}, {y:.1f})\n"
                )

        if self.geometry_calibrator.current_calibration:
            cal = self.geometry_calibrator.current_calibration
            self.corners_text.insert(
                tk.END, f"\nCalibration Error: {cal.calibration_error:.2f} pixels\n"
            )
            self.corners_text.insert(
                tk.END,
                f"Table Dimensions: {cal.table_dimensions_real[0]:.2f} x {cal.table_dimensions_real[1]:.2f} m\n",
            )

    def validate_camera(self) -> None:
        """Validate camera calibration."""
        if not self.calibration_images:
            messagebox.showwarning("Warning", "No calibration images available")
            return

        self.validation_results_text.insert(
            tk.END, "Validating camera calibration...\n"
        )
        result = self.validator.validate_camera_calibration(
            self.camera_calibrator, self.calibration_images
        )
        self.display_validation_result(result)

    def validate_colors(self) -> None:
        """Validate color calibration."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return

        # Create simple ground truth for demonstration
        ground_truth = {}
        if self.color_samples:
            for ball_type, regions in self.color_samples.items():
                ground_truth[ball_type] = regions

        if ground_truth:
            self.validation_results_text.insert(
                tk.END, "Validating color calibration...\n"
            )
            result = self.validator.validate_color_calibration(
                self.color_calibrator, self.current_frame, ground_truth
            )
            self.display_validation_result(result)
        else:
            messagebox.showwarning(
                "Warning", "No color samples available for validation"
            )

    def validate_geometry(self) -> None:
        """Validate geometric calibration."""
        if len(self.selected_corners) != 4:
            messagebox.showwarning("Warning", "No geometric calibration available")
            return

        # Create test points for validation
        test_points = self.selected_corners
        # Expected world points (table corners)
        w, h = self.table_width_var.get(), self.table_height_var.get()
        expected_points = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]

        self.validation_results_text.insert(
            tk.END, "Validating geometric calibration...\n"
        )
        result = self.validator.validate_geometric_calibration(
            self.geometry_calibrator, test_points, expected_points
        )
        self.display_validation_result(result)

    def full_validation(self) -> None:
        """Perform full system validation."""
        self.validation_results_text.insert(tk.END, "Starting full validation...\n")

        # Prepare test data
        test_data = {}
        if self.calibration_images:
            test_data["camera_test_images"] = self.calibration_images
        if self.current_frame is not None:
            test_data["color_test_frame"] = self.current_frame
        if self.color_samples:
            test_data["color_ground_truth"] = self.color_samples
        if len(self.selected_corners) == 4:
            test_data["geometry_test_points"] = self.selected_corners
            w, h = self.table_width_var.get(), self.table_height_var.get()
            test_data["geometry_expected_points"] = [
                (-w / 2, -h / 2),
                (w / 2, -h / 2),
                (w / 2, h / 2),
                (-w / 2, h / 2),
            ]

        if test_data:
            report = self.validator.generate_comprehensive_report(
                self.camera_calibrator,
                self.color_calibrator,
                self.geometry_calibrator,
                test_data,
            )
            self.display_validation_report(report)
        else:
            messagebox.showwarning("Warning", "No data available for validation")

    def display_validation_result(self, result) -> None:
        """Display validation result in text widget."""
        self.validation_results_text.insert(
            tk.END, f"\n=== {result.test_name.upper()} ===\n"
        )
        self.validation_results_text.insert(
            tk.END, f"Status: {'PASSED' if result.passed else 'FAILED'}\n"
        )
        self.validation_results_text.insert(
            tk.END, f"Accuracy Score: {result.accuracy_score:.3f}\n"
        )

        self.validation_results_text.insert(tk.END, "\nError Metrics:\n")
        for metric, value in result.error_metrics.items():
            if isinstance(value, (int, float)):
                self.validation_results_text.insert(
                    tk.END, f"  {metric}: {value:.3f}\n"
                )
            else:
                self.validation_results_text.insert(tk.END, f"  {metric}: {value}\n")

        self.validation_results_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.validation_results_text.see(tk.END)

    def display_validation_report(self, report) -> None:
        """Display comprehensive validation report."""
        self.validation_results_text.insert(
            tk.END, "\n=== COMPREHENSIVE VALIDATION REPORT ===\n"
        )
        self.validation_results_text.insert(
            tk.END, f"Session ID: {report.session_id}\n"
        )
        self.validation_results_text.insert(
            tk.END, f"Overall Score: {report.overall_score:.3f}\n"
        )

        self.validation_results_text.insert(tk.END, "\nRecommendations:\n")
        for rec in report.recommendations:
            self.validation_results_text.insert(tk.END, f"  • {rec}\n")

        self.validation_results_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.validation_results_text.see(tk.END)

    def update_camera_status(self, message) -> None:
        """Update camera status display."""
        self.camera_status_text.insert(tk.END, f"{message}\n")
        self.camera_status_text.see(tk.END)

    def __del__(self) -> None:
        """Cleanup when GUI is destroyed."""
        self.stop_camera()


def run_calibration_gui():
    """Run the interactive calibration GUI."""
    root = tk.Tk()
    app = InteractiveCalibrationGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop_camera()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run GUI
    run_calibration_gui()
