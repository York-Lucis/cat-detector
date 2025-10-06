"""
Unit tests for the utils module.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from utils import (
    calculate_color_distance, extract_roi_color, create_timestamp_filename,
    validate_image_file, get_file_size_mb, format_duration, safe_filename
)


class TestColorUtils(unittest.TestCase):
    """Test cases for color utility functions."""
    
    def test_calculate_color_distance(self):
        """Test color distance calculation."""
        color1 = np.array([100, 50, 200])
        color2 = np.array([110, 60, 210])
        
        distance = calculate_color_distance(color1, color2)
        
        # Should be approximately 17.32 (sqrt of 300)
        self.assertAlmostEqual(distance, 17.32, places=1)
    
    def test_calculate_color_distance_identical(self):
        """Test color distance with identical colors."""
        color = np.array([100, 50, 200])
        distance = calculate_color_distance(color, color)
        self.assertEqual(distance, 0.0)
    
    def test_extract_roi_color(self):
        """Test ROI color extraction."""
        # Create a test image with known color
        test_color = (100, 150, 200)  # BGR
        frame = np.full((100, 100, 3), test_color, dtype=np.uint8)
        
        roi_color = extract_roi_color(frame, 10, 10, 50, 50)
        
        self.assertIsNotNone(roi_color)
        self.assertEqual(len(roi_color), 3)
    
    def test_extract_roi_color_invalid_roi(self):
        """Test ROI color extraction with invalid region."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Invalid ROI (x2 < x1)
        roi_color = extract_roi_color(frame, 50, 10, 10, 50)
        self.assertIsNone(roi_color)
        
        # Empty ROI
        roi_color = extract_roi_color(frame, 10, 10, 10, 10)
        self.assertIsNone(roi_color)


class TestFileUtils(unittest.TestCase):
    """Test cases for file utility functions."""
    
    def test_create_timestamp_filename(self):
        """Test timestamp filename creation."""
        filename = create_timestamp_filename("test", "mp4")
        
        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".mp4"))
        self.assertIn("_", filename)
    
    def test_validate_image_file(self):
        """Test image file validation."""
        # Test with valid extension
        with patch('os.path.splitext', return_value=('test', '.jpg')):
            with patch('cv2.imread', return_value=np.array([[[0, 0, 0]]])):
                self.assertTrue(validate_image_file("test.jpg"))
        
        # Test with invalid extension
        with patch('os.path.splitext', return_value=('test', '.txt')):
            self.assertFalse(validate_image_file("test.txt"))
        
        # Test with None image
        with patch('os.path.splitext', return_value=('test', '.jpg')):
            with patch('cv2.imread', return_value=None):
                self.assertFalse(validate_image_file("test.jpg"))
    
    def test_get_file_size_mb(self):
        """Test file size calculation."""
        with patch('os.path.getsize', return_value=1048576):  # 1MB
            size = get_file_size_mb("test.txt")
            self.assertEqual(size, 1.0)
        
        with patch('os.path.getsize', side_effect=OSError()):
            size = get_file_size_mb("nonexistent.txt")
            self.assertEqual(size, 0.0)
    
    def test_safe_filename(self):
        """Test safe filename creation."""
        # Test with invalid characters
        unsafe_name = "test<>file|name?.txt"
        safe_name = safe_filename(unsafe_name)
        self.assertEqual(safe_name, "test_file_name_.txt")
        
        # Test with multiple underscores
        name_with_underscores = "test___file____name"
        safe_name = safe_filename(name_with_underscores)
        self.assertEqual(safe_name, "test_file_name")
        
        # Test with leading/trailing underscores
        name_with_edges = "_test_file_"
        safe_name = safe_filename(name_with_edges)
        self.assertEqual(safe_name, "test_file")


class TestFormatUtils(unittest.TestCase):
    """Test cases for formatting utility functions."""
    
    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        self.assertEqual(format_duration(30.5), "30.5s")
        self.assertEqual(format_duration(59.9), "59.9s")
    
    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        self.assertEqual(format_duration(90), "1m 30.0s")
        self.assertEqual(format_duration(125.5), "2m 5.5s")
    
    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        self.assertEqual(format_duration(3661), "1h 1m 1.0s")
        self.assertEqual(format_duration(7325.5), "2h 2m 5.5s")


if __name__ == '__main__':
    unittest.main()
