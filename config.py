"""
Configuration module for the Cat Detector application.

This module centralizes all constants, settings, and configuration values
used throughout the application. It provides a single source of truth for
all configuration parameters and ensures consistency across the codebase.

Author: Cat Detector Team
Version: 2.0.0
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Application paths
BASE_DIR = Path(__file__).parent
PROFILES_FOLDER = BASE_DIR / "cat_profiles"
VIDEO_RECORDINGS_FOLDER = BASE_DIR / "video_recordings"
MODEL_PATH = BASE_DIR / "yolov8n.pt"

# File naming constants
PROFILE_FILENAME = "profile.npy"
PROFILE_METADATA_FILENAME = "profile.json"

# Detection and recognition settings
COLOR_MATCH_TOLERANCE = 35
CONFIDENCE_THRESHOLD = 0.65
CAT_CLASS_ID = 15  # YOLOv8 COCO class for 'cat'

# Video recording settings
MAX_RECORDING_DURATION = 60  # Maximum recording duration in seconds
RECORDING_COOLDOWN = 5      # Seconds to wait after cat disappears before stopping recording
VIDEO_FPS = 20.0
VIDEO_CODEC = 'mp4v'

# UI settings
WINDOW_TITLE = "Cat Recognition Tool"
WINDOW_SIZE = "1000x750"
PROFILE_MANAGER_SIZE = "400x350"

# Font settings
DEFAULT_FONT_SCALE = 0.7
DEFAULT_FONT_SIZE = 20

# Color settings (BGR format for OpenCV)
COLORS = {
    'GENERAL_DETECTION': (0, 255, 0),    # Green
    'SPECIFIC_DETECTION': (0, 0, 255),   # Red
    'TEXT_COLOR': (0, 255, 0),           # Green
    'BACKGROUND': 'black'
}

# Threading settings
THREAD_DAEMON = True
VIDEO_LOOP_DELAY = 10  # milliseconds

# Camera settings
MAX_CAMERA_INDEX = 10
CAMERA_BACKEND = 'CAP_DSHOW'  # Windows DirectShow backend

# Profile creation settings
MIN_PROFILE_IMAGES = 2
MAX_PROFILE_IMAGES = 4
KMEANS_CLUSTERS = 3
KMEANS_RANDOM_STATE = 0

# Status messages
STATUS_MESSAGES = {
    'INITIALIZING': "Status: Initializing...",
    'LOADING_MODEL': "Status: Loading YOLOv8 model onto {device}...",
    'MODEL_LOADED': "Status: Model loaded on {device}. Ready.",
    'MODEL_ERROR': "Status: Error loading model. Check internet.",
    'DETECTING': "Status: Detecting... | Profile: {profile} | Device: {device}",
    'RECORDING': "Status: RECORDING... | Duration: {duration:.1f}s",
    'RECORDING_ERROR': "Status: Recording error",
    'READY': "Status: Ready"
}

# Error messages
ERROR_MESSAGES = {
    'NO_WEBCAMS': "No Webcams Found",
    'MODEL_LOAD_FAILED': "Failed to load YOLOv8 model: {error}",
    'WEBCAM_ERROR': "Could not open {webcam}.",
    'INCOMPLETE_DATA': "All fields are required.",
    'PROFILE_EXISTS': "A profile named '{name}' already exists.",
    'NO_SELECTION': "Please select a profile to delete.",
    'CONFIRM_DELETE': "Are you sure you want to delete the profile '{name}'?",
    'PROFILE_CREATION_FAILED': "Could not create a color profile for '{name}'. Please use clearer images.",
    'PROFILE_CREATED': "Profile '{name}' was created successfully.",
    'MODEL_NOT_LOADED': "Model is not loaded yet."
}

# Ensure required directories exist
def ensure_directories() -> None:
    """
    Create required directories if they don't exist.
    
    This function ensures that all necessary directories for the application
    are created before the application starts. It's safe to call multiple times.
    
    Raises:
        OSError: If directory creation fails due to permission issues.
    """
    PROFILES_FOLDER.mkdir(exist_ok=True)
    VIDEO_RECORDINGS_FOLDER.mkdir(exist_ok=True)

# Font fallback options
FONT_FALLBACKS = [
    "arial.ttf",
    "DejaVuSans.ttf"
]
