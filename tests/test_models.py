"""
Unit tests for the models module.
"""

import unittest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from models import CatProfile, DetectionResult, VideoRecording, ApplicationState, ProfileManager


class TestCatProfile(unittest.TestCase):
    """Test cases for CatProfile class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profile_data = {
            'profile_name': 'test_cat',
            'display_name': 'Test Cat',
            'breed': 'Persian'
        }
    
    def test_profile_creation(self):
        """Test profile creation with basic data."""
        profile = CatProfile(**self.profile_data)
        
        self.assertEqual(profile.profile_name, 'test_cat')
        self.assertEqual(profile.display_name, 'Test Cat')
        self.assertEqual(profile.breed, 'Persian')
        self.assertIsNotNone(profile.created_at)
        self.assertIsNotNone(profile.updated_at)
    
    def test_profile_to_dict(self):
        """Test profile serialization to dictionary."""
        profile = CatProfile(**self.profile_data)
        profile_dict = profile.to_dict()
        
        self.assertEqual(profile_dict['profile_name'], 'test_cat')
        self.assertEqual(profile_dict['display_name'], 'Test Cat')
        self.assertEqual(profile_dict['breed'], 'Persian')
        self.assertIn('created_at', profile_dict)
        self.assertIn('updated_at', profile_dict)
    
    def test_profile_from_dict(self):
        """Test profile creation from dictionary."""
        profile_dict = {
            'profile_name': 'test_cat',
            'display_name': 'Test Cat',
            'breed': 'Persian',
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00'
        }
        
        profile = CatProfile.from_dict(profile_dict)
        
        self.assertEqual(profile.profile_name, 'test_cat')
        self.assertEqual(profile.display_name, 'Test Cat')
        self.assertEqual(profile.breed, 'Persian')
        self.assertIsInstance(profile.created_at, datetime)
        self.assertIsInstance(profile.updated_at, datetime)
    
    def test_update_timestamp(self):
        """Test timestamp update functionality."""
        profile = CatProfile(**self.profile_data)
        original_updated = profile.updated_at
        
        # Wait a small amount to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        profile.update_timestamp()
        
        self.assertGreater(profile.updated_at, original_updated)


class TestDetectionResult(unittest.TestCase):
    """Test cases for DetectionResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detection = DetectionResult(
            confidence=0.85,
            bounding_box=(100, 100, 200, 200)
        )
    
    def test_detection_creation(self):
        """Test detection result creation."""
        self.assertEqual(self.detection.confidence, 0.85)
        self.assertEqual(self.detection.bounding_box, (100, 100, 200, 200))
        self.assertFalse(self.detection.is_specific_cat)
        self.assertIsNone(self.detection.cat_name)
    
    def test_bounding_box_properties(self):
        """Test bounding box property access."""
        self.assertEqual(self.detection.x1, 100)
        self.assertEqual(self.detection.y1, 100)
        self.assertEqual(self.detection.x2, 200)
        self.assertEqual(self.detection.y2, 200)
        self.assertEqual(self.detection.width, 100)
        self.assertEqual(self.detection.height, 100)
    
    def test_specific_cat_detection(self):
        """Test specific cat detection properties."""
        self.detection.is_specific_cat = True
        self.detection.cat_name = "Fluffy"
        self.detection.color_distance = 25.5
        
        self.assertTrue(self.detection.is_specific_cat)
        self.assertEqual(self.detection.cat_name, "Fluffy")
        self.assertEqual(self.detection.color_distance, 25.5)


class TestVideoRecording(unittest.TestCase):
    """Test cases for VideoRecording class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_time = datetime.now()
        self.recording = VideoRecording(
            filename="test_video.mp4",
            start_time=self.start_time,
            cat_name="Test Cat"
        )
    
    def test_recording_creation(self):
        """Test video recording creation."""
        self.assertEqual(self.recording.filename, "test_video.mp4")
        self.assertEqual(self.recording.start_time, self.start_time)
        self.assertEqual(self.recording.cat_name, "Test Cat")
        self.assertTrue(self.recording.is_active)
        self.assertEqual(self.recording.duration, 0.0)
    
    def test_duration_update(self):
        """Test duration update functionality."""
        current_time = datetime.now()
        self.recording.update_duration(current_time)
        
        expected_duration = (current_time - self.start_time).total_seconds()
        self.assertAlmostEqual(self.recording.duration, expected_duration, places=2)


class TestApplicationState(unittest.TestCase):
    """Test cases for ApplicationState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = ApplicationState()
    
    def test_initial_state(self):
        """Test initial application state."""
        self.assertFalse(self.state.is_running)
        self.assertFalse(self.state.is_recording)
        self.assertIsNone(self.state.active_profile)
        self.assertIsNone(self.state.selected_webcam)
        self.assertEqual(self.state.device, 'cpu')
        self.assertFalse(self.state.model_loaded)
    
    def test_reset_recording_state(self):
        """Test recording state reset."""
        self.state.is_recording = True
        self.state.current_recording = VideoRecording(
            filename="test.mp4",
            start_time=datetime.now(),
            cat_name="Test"
        )
        self.state.last_detection_time = datetime.now()
        
        self.state.reset_recording_state()
        
        self.assertFalse(self.state.is_recording)
        self.assertIsNone(self.state.current_recording)
        self.assertIsNone(self.state.last_detection_time)


class TestProfileManager(unittest.TestCase):
    """Test cases for ProfileManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_manager = ProfileManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_profile_manager_creation(self):
        """Test profile manager creation."""
        self.assertEqual(self.profile_manager.profiles_folder, Path(self.temp_dir))
        self.assertIsInstance(self.profile_manager.profiles, dict)
    
    def test_load_empty_profiles(self):
        """Test loading profiles from empty directory."""
        profiles = self.profile_manager.load_profiles()
        self.assertEqual(len(profiles), 0)
    
    def test_profile_operations(self):
        """Test basic profile operations."""
        # Create a test profile
        profile = CatProfile(
            profile_name="test_cat",
            display_name="Test Cat",
            breed="Persian",
            signature=np.array([100, 50, 200])
        )
        
        # Test save
        success = self.profile_manager.save_profile(profile)
        self.assertTrue(success)
        
        # Test get
        retrieved_profile = self.profile_manager.get_profile("test_cat")
        self.assertIsNotNone(retrieved_profile)
        self.assertEqual(retrieved_profile.profile_name, "test_cat")
        
        # Test list
        profiles = self.profile_manager.list_profiles()
        self.assertIn("test_cat", profiles)
        
        # Test exists
        self.assertTrue(self.profile_manager.profile_exists("test_cat"))
        self.assertFalse(self.profile_manager.profile_exists("nonexistent"))
        
        # Test delete
        success = self.profile_manager.delete_profile("test_cat")
        self.assertTrue(success)
        self.assertFalse(self.profile_manager.profile_exists("test_cat"))


if __name__ == '__main__':
    unittest.main()
