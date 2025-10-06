"""
Utility functions for the Cat Detector application.

This module contains common helper functions used across the application,
including image processing, file operations, text rendering, and various
utility functions for data manipulation and validation.

Author: Cat Detector Team
Version: 2.0.0
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional, Union
import os
import sys
import locale
import codecs
from pathlib import Path
from config import FONT_FALLBACKS, DEFAULT_FONT_SCALE, DEFAULT_FONT_SIZE


def setup_utf8_encoding() -> None:
    """
    Set up UTF-8 encoding for the application.
    
    This function configures the system to use UTF-8 encoding for text output,
    which is essential for proper handling of Unicode characters in the application.
    Different approaches are used for Windows and Unix-like systems.
    
    Raises:
        Exception: If encoding setup fails (logged as warning)
    """
    try:
        if sys.platform.startswith('win'):
            # On Windows, try to set UTF-8 mode
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        else:
            # On Unix-like systems, set locale
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")


def draw_unicode_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
                     font_scale: float = DEFAULT_FONT_SCALE, 
                     color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
    """
    Draw text with Unicode support using PIL instead of OpenCV's putText.
    This handles special characters like Ç, ñ, é, etc.
    
    Args:
        frame: Input frame as numpy array
        text: Text to draw
        position: (x, y) position for text
        font_scale: Scale factor for font size
        color: BGR color tuple
        thickness: Text thickness (not used in PIL, kept for compatibility)
        
    Returns:
        Frame with text drawn
    """
    try:
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a system font that supports Unicode
        font = _get_unicode_font(font_scale)
        
        # Convert BGR color to RGB for PIL
        rgb_color = (color[2], color[1], color[0])
        
        # Draw text
        draw.text(position, text, font=font, fill=rgb_color)
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        # Fallback to OpenCV putText if PIL fails
        print(f"Unicode text drawing failed, using fallback: {e}")
        cv2.putText(frame, text.encode('ascii', 'replace').decode('ascii'), position, 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return frame


def _get_unicode_font(font_scale: float) -> ImageFont.FreeTypeFont:
    """Get a Unicode-compatible font."""
    font_size = int(font_scale * DEFAULT_FONT_SIZE)
    
    for font_name in FONT_FALLBACKS:
        try:
            return ImageFont.truetype(font_name, font_size)
        except (OSError, IOError):
            continue
    
    # Fallback to default font
    return ImageFont.load_default()


def detect_available_webcams(max_index: int = 10) -> List[int]:
    """
    Detect available webcam devices.
    
    Args:
        max_index: Maximum camera index to check
        
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras


def extract_dominant_colors(image: np.ndarray, n_clusters: int = 3) -> Optional[np.ndarray]:
    """
    Extract dominant colors from an image using K-Means clustering.
    
    Args:
        image: Input image as numpy array
        n_clusters: Number of color clusters to find
        
    Returns:
        Dominant color as HSV array, or None if extraction fails
    """
    try:
        from sklearn.cluster import KMeans
        
        # Convert to HSV for better color analysis
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixels = img_hsv.reshape(-1, 3)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
        kmeans.fit(pixels)
        
        # Get the most dominant color
        counts = np.bincount(kmeans.labels_)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
        
        return dominant_color
    except Exception as e:
        print(f"Color extraction failed: {e}")
        return None


def calculate_color_distance(color1: np.ndarray, color2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two colors in HSV space.
    
    Args:
        color1: First color as HSV array
        color2: Second color as HSV array
        
    Returns:
        Euclidean distance between the colors
    """
    return np.linalg.norm(color1 - color2)


def extract_roi_color(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """
    Extract average color from a region of interest.
    
    Args:
        frame: Input frame
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        Average color in HSV space, or None if extraction fails
    """
    try:
        # Extract region of interest
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Convert to HSV and calculate average color
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(roi_hsv.reshape(-1, 3), axis=0)
        
        return avg_color
    except Exception as e:
        print(f"ROI color extraction failed: {e}")
        return None


def create_timestamp_filename(base_name: str, extension: str = "mp4") -> str:
    """
    Create a filename with timestamp.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        
    Returns:
        Filename with timestamp
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"


def validate_image_file(filepath: str) -> bool:
    """
    Validate if a file is a valid image.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext not in valid_extensions:
            return False
        
        # Try to load the image
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name
