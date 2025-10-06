"""
Main application file for the Cat Detector application.

This module contains the main application class and entry point for the
refactored Cat Detector application. It demonstrates clean code principles
with proper separation of concerns, dependency injection, and modular architecture.

The application provides a GUI interface for real-time cat detection and
recognition using computer vision and machine learning techniques.

Author: Cat Detector Team
Version: 2.0.0
"""

import tkinter as tk
import cv2
import threading
import time
from typing import Optional, List, Tuple
import tkinter.messagebox as messagebox

from config import ensure_directories, WINDOW_TITLE, WINDOW_SIZE, VIDEO_LOOP_DELAY, THREAD_DAEMON
from models import DetectionResult
from services import ApplicationService
from ui_components import VideoDisplay, StatusBar, ControlPanel, ProfileManager
from utils import setup_utf8_encoding, draw_unicode_text, format_duration
from logger import logger


class CatDetectorApp:
    """
    Main application class for the Cat Detector.
    
    This class orchestrates the entire application, managing the GUI components,
    service layer, and application lifecycle. It follows the MVC pattern with
    clear separation between UI, business logic, and data models.
    
    Attributes:
        root: Main Tkinter root window
        app_service: Application service layer instance
        cap: OpenCV video capture instance
        video_display: Custom video display widget
        control_panel: Control panel with webcam and profile selection
        status_bar: Status bar for displaying application state
    """
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the Cat Detector application.
        
        Args:
            root: Tkinter root window instance
        """
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        
        # Initialize services
        self.app_service = ApplicationService()
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Create UI components
        self._create_ui()
        
        # Setup application
        self._setup_application()
        
        # Initialize backend in separate thread
        self._initialize_backend()
    
    def _create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = tk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_display = VideoDisplay(
            main_frame, 
            text="Webcam feed will appear here.", 
            anchor=tk.CENTER, 
            background='black'
        )
        self.video_display.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Control panel
        self.control_panel = ControlPanel(
            main_frame,
            on_webcam_change=self._on_webcam_change,
            on_profile_change=self._on_profile_change
        )
        self.control_panel.pack(fill=tk.X)
        
        # Set button callbacks
        self.control_panel.set_webcam_toggle_callback(self._toggle_webcam)
        self.control_panel.set_profile_manage_callback(self._open_profile_manager)
        
        # Status bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_application(self):
        """Setup application directories and initial state."""
        ensure_directories()
        logger.info("Application directories created")
    
    def _initialize_backend(self):
        """Initialize backend services in a separate thread."""
        def init_thread():
            try:
                self.status_bar.set_status("Initializing services...")
                
                if not self.app_service.initialize():
                    self.status_bar.set_status("Failed to initialize services")
                    return
                
                # Populate UI with available options
                self._populate_webcams()
                self._populate_profiles()
                
                self.status_bar.set_status("Ready")
                logger.info("Backend initialization completed")
                
            except Exception as e:
                logger.error(f"Backend initialization failed: {e}")
                self.status_bar.set_status("Initialization failed")
        
        thread = threading.Thread(target=init_thread, daemon=THREAD_DAEMON)
        thread.start()
    
    def _populate_webcams(self):
        """Populate webcam dropdown with available cameras."""
        try:
            webcams = self.app_service.get_available_webcams()
            self.control_panel.populate_webcams(webcams)
            logger.info(f"Found {len(webcams)} webcams")
        except Exception as e:
            logger.error(f"Failed to populate webcams: {e}")
    
    def _populate_profiles(self):
        """Populate profile dropdown with available profiles."""
        try:
            profiles = self.app_service.profile_service.list_profiles()
            self.control_panel.populate_profiles(profiles)
            logger.info(f"Loaded {len(profiles)} profiles")
        except Exception as e:
            logger.error(f"Failed to populate profiles: {e}")
    
    def _on_webcam_change(self, webcam_name: str):
        """Handle webcam selection change."""
        logger.info(f"Webcam changed to: {webcam_name}")
    
    def _on_profile_change(self, profile_name: str):
        """Handle profile selection change."""
        self.app_service.set_active_profile(profile_name)
        logger.info(f"Profile changed to: {profile_name}")
    
    def _toggle_webcam(self):
        """Toggle webcam on/off."""
        if self.app_service.state.is_running:
            self._stop_webcam()
        else:
            self._start_webcam()
    
    def _start_webcam(self):
        """Start webcam capture and detection."""
        if not self.app_service.detection_service.is_model_loaded():
            tk.messagebox.showerror("Error", "Model is not loaded yet.")
            return
        
        # Get selected webcam
        webcam_index = self.control_panel.get_selected_webcam()
        if webcam_index is None:
            tk.messagebox.showerror("Webcam Error", "No webcam selected.")
            return
        
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise ValueError("Cannot open camera")
            
            # Update state
            self.app_service.state.is_running = True
            self.app_service.state.selected_webcam = webcam_index
            
            # Update UI
            self.control_panel.set_webcam_button_text("Stop Webcam")
            self.status_bar.set_detection_status(
                self.control_panel.get_selected_profile(),
                self.app_service.state.device
            )
            
            # Start video loop
            self._video_loop()
            
            logger.info(f"Webcam started: {webcam_index}")
            
        except Exception as e:
            logger.error(f"Failed to start webcam: {e}")
            tk.messagebox.showerror("Webcam Error", f"Could not open webcam {webcam_index}.")
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def _stop_webcam(self):
        """Stop webcam capture."""
        self.app_service.state.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Stop any active recording
        self.app_service.recording_service.stop_recording()
        
        # Update UI
        self.control_panel.set_webcam_button_text("Start Webcam")
        self.video_display.show_placeholder("Webcam feed stopped.")
        self.status_bar.set_status("Webcam stopped")
        
        logger.info("Webcam stopped")
    
    def _video_loop(self):
        """Main video processing loop."""
        if not self.app_service.state.is_running:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                self.root.after(VIDEO_LOOP_DELAY, self._video_loop)
                return
            
            # Process frame for detection
            detections, should_record = self.app_service.process_frame(frame)
            
            # Draw detections on frame
            frame = self._draw_detections(frame, detections)
            
            # Update video display
            self.video_display.update_frame(frame)
            
            # Update status if recording
            if self.app_service.recording_service.is_recording:
                duration = self.app_service.recording_service.get_recording_duration()
                self.status_bar.set_recording_status(duration)
            
            # Schedule next frame
            self.root.after(VIDEO_LOOP_DELAY, self._video_loop)
            
        except Exception as e:
            logger.error(f"Video loop error: {e}")
            self.root.after(VIDEO_LOOP_DELAY, self._video_loop)
    
    def _draw_detections(self, frame, detections: list[DetectionResult]) -> cv2.Mat:
        """Draw detection results on the frame."""
        for detection in detections:
            x1, y1, x2, y2 = detection.bounding_box
            
            # Determine color and text based on detection type
            if detection.is_specific_cat:
                color = (0, 0, 255)  # Red for specific cat
                text = f"!!! {detection.cat_name} !!!"
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for general cat
                text = f"Cat: {detection.confidence:.2f}"
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw text
            frame = draw_unicode_text(frame, text, (x1, y1 - 10), 0.7, color, 2)
        
        return frame
    
    def _open_profile_manager(self):
        """Open the profile management window."""
        self.control_panel.profile_button.config(state="disabled")
        
        manager = ProfileManager(
            self,
            self.app_service.profile_service,
            on_profile_change=self._on_profiles_changed
        )
        
        self.root.wait_window(manager)
        self.control_panel.profile_button.config(state="normal")
    
    def _on_profiles_changed(self):
        """Handle profile changes from profile manager."""
        self._populate_profiles()
        logger.info("Profiles updated")
    
    def _on_closing(self):
        """Handle application closing."""
        logger.info("Application closing")
        
        # Stop webcam
        if self.app_service.state.is_running:
            self._stop_webcam()
        
        # Cleanup services
        self.app_service.cleanup()
        
        # Close application
        self.root.destroy()


def main():
    """Main entry point for the application."""
    # Setup UTF-8 encoding
    setup_utf8_encoding()
    
    # Create and run application
    root = tk.Tk()
    app = CatDetectorApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
