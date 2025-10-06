"""
Logging configuration for the Cat Detector application.
Provides centralized logging functionality with different levels and handlers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from config import BASE_DIR


class CatDetectorLogger:
    """Centralized logger for the Cat Detector application."""
    
    def __init__(self, name: str = "cat_detector", log_level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"cat_detector_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback."""
        self.logger.exception(message)


# Global logger instance
logger = CatDetectorLogger()


class ErrorHandler:
    """Centralized error handling for the application."""
    
    @staticmethod
    def handle_camera_error(error: Exception, camera_index: int) -> str:
        """Handle camera-related errors."""
        error_msg = f"Camera {camera_index} error: {str(error)}"
        logger.error(error_msg)
        return error_msg
    
    @staticmethod
    def handle_model_error(error: Exception) -> str:
        """Handle model loading/processing errors."""
        error_msg = f"Model error: {str(error)}"
        logger.error(error_msg)
        return error_msg
    
    @staticmethod
    def handle_profile_error(error: Exception, profile_name: str) -> str:
        """Handle profile-related errors."""
        error_msg = f"Profile '{profile_name}' error: {str(error)}"
        logger.error(error_msg)
        return error_msg
    
    @staticmethod
    def handle_video_error(error: Exception, operation: str) -> str:
        """Handle video recording/processing errors."""
        error_msg = f"Video {operation} error: {str(error)}"
        logger.error(error_msg)
        return error_msg
    
    @staticmethod
    def handle_file_error(error: Exception, filepath: str) -> str:
        """Handle file I/O errors."""
        error_msg = f"File error for '{filepath}': {str(error)}"
        logger.error(error_msg)
        return error_msg


def log_function_call(func_name: str, **kwargs):
    """Log function call with parameters."""
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")


def log_performance(func_name: str, duration: float):
    """Log function performance."""
    logger.debug(f"{func_name} completed in {duration:.3f}s")


def log_detection_result(confidence: float, is_specific: bool, cat_name: Optional[str] = None):
    """Log detection results."""
    if is_specific and cat_name:
        logger.info(f"Detected specific cat: {cat_name} (confidence: {confidence:.2f})")
    else:
        logger.info(f"Detected general cat (confidence: {confidence:.2f})")


def log_recording_event(event: str, cat_name: str, filename: Optional[str] = None):
    """Log video recording events."""
    if filename:
        logger.info(f"Recording {event}: {cat_name} -> {filename}")
    else:
        logger.info(f"Recording {event}: {cat_name}")


def log_profile_operation(operation: str, profile_name: str, success: bool):
    """Log profile management operations."""
    status = "success" if success else "failed"
    logger.info(f"Profile {operation} {status}: {profile_name}")
