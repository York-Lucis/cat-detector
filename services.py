"""
Service layer for the Cat Detector application.

This module contains the business logic for the application, including
detection services, profile management, video recording, and application
coordination. It follows the service layer pattern to separate business
logic from UI and data access concerns.

Author: Cat Detector Team
Version: 2.0.0
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import shutil
import time
from datetime import datetime

from models import CatProfile, DetectionResult, VideoRecording, ApplicationState, ProfileManager
from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, CAT_CLASS_ID, COLOR_MATCH_TOLERANCE,
    MAX_RECORDING_DURATION, RECORDING_COOLDOWN, VIDEO_FPS, VIDEO_CODEC,
    VIDEO_RECORDINGS_FOLDER, KMEANS_CLUSTERS
)
from utils import extract_dominant_colors, calculate_color_distance, extract_roi_color, create_timestamp_filename
from logger import logger, log_detection_result, log_recording_event, log_profile_operation


class DetectionService:
    """
    Service for cat detection and recognition.
    
    This service handles the YOLO model initialization, cat detection in frames,
    and specific cat recognition using color matching. It provides a clean
    interface for the detection functionality while managing model state.
    
    Attributes:
        model: YOLO model instance for object detection
        device: Device used for model inference ('cpu' or 'cuda')
        _model_loaded: Internal flag indicating if model is ready for use
    """
    
    def __init__(self) -> None:
        """Initialize the detection service."""
        self.model: Optional[YOLO] = None
        self.device: str = 'cpu'
        self._model_loaded: bool = False
    
    def initialize_model(self) -> bool:
        """Initialize the YOLO model."""
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing YOLO model on {self.device.upper()}")
            
            self.model = YOLO(str(MODEL_PATH))
            self.model.to(self.device)
            self._model_loaded = True
            
            logger.info(f"YOLO model loaded successfully on {self.device.upper()}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            self._model_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model_loaded and self.model is not None
    
    def detect_cats(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect cats in a frame and return detection results.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of DetectionResult objects
        """
        if not self.is_model_loaded():
            logger.warning("Model not loaded, cannot perform detection")
            return []
        
        try:
            results = self.model(frame, verbose=False, device=self.device)
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    if int(box.cls[0]) == CAT_CLASS_ID and float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        detection = DetectionResult(
                            confidence=confidence,
                            bounding_box=(x1, y1, x2, y2)
                        )
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def recognize_cat(self, frame: np.ndarray, detection: DetectionResult, 
                     profile: CatProfile) -> Optional[DetectionResult]:
        """
        Attempt to recognize a specific cat from a detection.
        
        Args:
            frame: Input frame
            detection: Detection result to analyze
            profile: Cat profile to match against
            
        Returns:
            Updated DetectionResult with recognition info, or None if no match
        """
        if not profile.signature:
            return None
        
        try:
            # Extract color from the detected region
            roi_color = extract_roi_color(
                frame, detection.x1, detection.y1, detection.x2, detection.y2
            )
            
            if roi_color is None:
                return None
            
            # Calculate color distance
            color_distance = calculate_color_distance(roi_color, profile.signature)
            
            # Check if it matches the profile
            if color_distance < COLOR_MATCH_TOLERANCE:
                detection.is_specific_cat = True
                detection.cat_name = profile.display_name
                detection.color_distance = color_distance
                
                log_detection_result(detection.confidence, True, profile.display_name)
                return detection
        
        except Exception as e:
            logger.error(f"Cat recognition failed: {e}")
        
        return None


class ProfileService:
    """Service for cat profile management."""
    
    def __init__(self, profile_manager: ProfileManager):
        self.profile_manager = profile_manager
    
    def create_profile(self, profile_data: Dict[str, Any], image_paths: List[str]) -> bool:
        """
        Create a new cat profile from provided data and images.
        
        Args:
            profile_data: Profile metadata
            image_paths: List of image file paths
            
        Returns:
            True if profile was created successfully
        """
        try:
            profile_name = profile_data['profile_name']
            logger.info(f"Creating profile: {profile_name}")
            
            # Create profile object
            profile = CatProfile(
                profile_name=profile_name,
                display_name=profile_data['display_name'],
                breed=profile_data['breed'],
                image_paths=image_paths
            )
            
            # Process images and create color signature
            signature = self._create_color_signature(image_paths)
            if signature is None:
                logger.error(f"Failed to create color signature for {profile_name}")
                return False
            
            profile.signature = signature
            
            # Save profile
            success = self.profile_manager.save_profile(profile)
            log_profile_operation("creation", profile_name, success)
            
            return success
        except Exception as e:
            logger.error(f"Profile creation failed: {e}")
            return False
    
    def _create_color_signature(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """
        Create a color signature from multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Average color signature as numpy array, or None if failed
        """
        try:
            all_dominant_colors = []
            
            for image_path in image_paths:
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                dominant_color = extract_dominant_colors(img, KMEANS_CLUSTERS)
                if dominant_color is not None:
                    all_dominant_colors.append(dominant_color)
            
            if not all_dominant_colors:
                logger.error("No valid colors extracted from images")
                return None
            
            # Calculate average color signature
            avg_signature = np.mean(all_dominant_colors, axis=0)
            logger.info(f"Created color signature from {len(all_dominant_colors)} images")
            
            return avg_signature
        except Exception as e:
            logger.error(f"Color signature creation failed: {e}")
            return None
    
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a cat profile.
        
        Args:
            profile_name: Name of the profile to delete
            
        Returns:
            True if profile was deleted successfully
        """
        try:
            success = self.profile_manager.delete_profile(profile_name)
            log_profile_operation("deletion", profile_name, success)
            return success
        except Exception as e:
            logger.error(f"Profile deletion failed: {e}")
            return False
    
    def get_profile(self, profile_name: str) -> Optional[CatProfile]:
        """Get a profile by name."""
        return self.profile_manager.get_profile(profile_name)
    
    def list_profiles(self) -> List[str]:
        """Get list of all profile names."""
        return self.profile_manager.list_profiles()
    
    def load_profiles(self) -> Dict[str, CatProfile]:
        """Load all profiles from storage."""
        return self.profile_manager.load_profiles()


class VideoRecordingService:
    """Service for video recording functionality."""
    
    def __init__(self):
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_recording: Optional[VideoRecording] = None
        self.is_recording = False
        self.recording_start_time: Optional[float] = None
        self.last_seen_time: Optional[float] = None
    
    def start_recording(self, cat_name: str, frame: np.ndarray) -> bool:
        """
        Start recording a video for a specific cat.
        
        Args:
            cat_name: Name of the cat being recorded
            frame: Initial frame to determine video properties
            
        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        try:
            height, width, _ = frame.shape
            filename = create_timestamp_filename(cat_name, "mp4")
            filepath = VIDEO_RECORDINGS_FOLDER / filename
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            self.video_writer = cv2.VideoWriter(
                str(filepath), fourcc, VIDEO_FPS, (width, height)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Could not initialize video writer")
            
            # Create recording object
            self.current_recording = VideoRecording(
                filename=filename,
                start_time=datetime.now(),
                cat_name=cat_name
            )
            
            self.is_recording = True
            self.recording_start_time = time.time()
            
            log_recording_event("started", cat_name, filename)
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._cleanup_recording()
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop the current recording.
        
        Returns:
            True if recording was stopped successfully
        """
        if not self.is_recording or not self.video_writer:
            return False
        
        try:
            self.video_writer.release()
            self.video_writer = None
            
            if self.current_recording:
                log_recording_event("stopped", self.current_recording.cat_name, 
                                  self.current_recording.filename)
            
            self._cleanup_recording()
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self._cleanup_recording()
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the current recording.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if frame was written successfully
        """
        if not self.is_recording or not self.video_writer:
            return False
        
        try:
            self.video_writer.write(frame)
            
            # Check for maximum duration
            if self.recording_start_time:
                elapsed_time = time.time() - self.recording_start_time
                if elapsed_time >= MAX_RECORDING_DURATION:
                    logger.info("Maximum recording duration reached")
                    self.stop_recording()
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            self.stop_recording()
            return False
    
    def update_last_seen(self):
        """Update the last seen timestamp."""
        self.last_seen_time = time.time()
    
    def should_stop_recording(self) -> bool:
        """
        Check if recording should be stopped based on cooldown.
        
        Returns:
            True if recording should be stopped
        """
        if not self.is_recording or not self.last_seen_time:
            return False
        
        time_since_last_seen = time.time() - self.last_seen_time
        return time_since_last_seen > RECORDING_COOLDOWN
    
    def get_recording_duration(self) -> float:
        """
        Get the current recording duration in seconds.
        
        Returns:
            Recording duration in seconds
        """
        if not self.is_recording or not self.recording_start_time:
            return 0.0
        
        return time.time() - self.recording_start_time
    
    def _cleanup_recording(self):
        """Clean up recording state."""
        self.is_recording = False
        self.current_recording = None
        self.recording_start_time = None
        self.last_seen_time = None


class ApplicationService:
    """Main application service that coordinates other services."""
    
    def __init__(self):
        self.profile_manager = ProfileManager(PROFILES_FOLDER)
        self.detection_service = DetectionService()
        self.profile_service = ProfileService(self.profile_manager)
        self.recording_service = VideoRecordingService()
        self.state = ApplicationState()
    
    def initialize(self) -> bool:
        """Initialize all services."""
        logger.info("Initializing application services")
        
        # Initialize detection service
        if not self.detection_service.initialize_model():
            return False
        
        # Load profiles
        self.profile_service.load_profiles()
        
        # Update state
        self.state.model_loaded = True
        self.state.device = self.detection_service.device
        
        logger.info("Application services initialized successfully")
        return True
    
    def get_available_webcams(self) -> List[int]:
        """Get list of available webcam indices."""
        from utils import detect_available_webcams
        return detect_available_webcams()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[DetectionResult], bool]:
        """
        Process a frame for detection and recording.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, should_record)
        """
        # Detect cats
        detections = self.detection_service.detect_cats(frame)
        
        # Check for specific cat recognition
        should_record = False
        if self.state.active_profile and detections:
            for detection in detections:
                recognized = self.detection_service.recognize_cat(
                    frame, detection, self.state.active_profile
                )
                if recognized and recognized.is_specific_cat:
                    should_record = True
                    self.recording_service.update_last_seen()
                    break
        
        # Handle recording logic
        if should_record and not self.recording_service.is_recording:
            cat_name = self.state.active_profile.display_name if self.state.active_profile else "Unknown"
            self.recording_service.start_recording(cat_name, frame)
        elif self.recording_service.should_stop_recording():
            self.recording_service.stop_recording()
        
        # Write frame if recording
        if self.recording_service.is_recording:
            self.recording_service.write_frame(frame)
        
        return detections, should_record
    
    def set_active_profile(self, profile_name: Optional[str]):
        """Set the active profile for recognition."""
        if profile_name is None or profile_name == "None (General Detection)":
            self.state.active_profile = None
        else:
            self.state.active_profile = self.profile_service.get_profile(profile_name)
        
        logger.info(f"Active profile set to: {profile_name}")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up application services")
        self.recording_service.stop_recording()
        self.state.reset_recording_state()
