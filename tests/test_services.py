"""
Unit tests for the services module.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from services import DetectionService, ProfileService, VideoRecordingService, ApplicationService
from models import CatProfile, ProfileManager


class TestDetectionService(unittest.TestCase):
    """Test cases for DetectionService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNone(self.detection_service.model)
        self.assertEqual(self.detection_service.device, 'cpu')
        self.assertFalse(self.detection_service._model_loaded)
    
    @patch('services.YOLO')
    @patch('services.torch')
    def test_initialize_model_success(self, mock_torch, mock_yolo):
        """Test successful model initialization."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        result = self.detection_service.initialize_model()
        
        self.assertTrue(result)
        self.assertTrue(self.detection_service._model_loaded)
        self.assertEqual(self.detection_service.device, 'cpu')
        mock_model.to.assert_called_once_with('cpu')
    
    @patch('services.YOLO')
    def test_initialize_model_failure(self, mock_yolo):
        """Test model initialization failure."""
        mock_yolo.side_effect = Exception("Model load failed")
        
        result = self.detection_service.initialize_model()
        
        self.assertFalse(result)
        self.assertFalse(self.detection_service._model_loaded)
    
    def test_is_model_loaded(self):
        """Test model loaded status check."""
        self.assertFalse(self.detection_service.is_model_loaded())
        
        self.detection_service._model_loaded = True
        self.detection_service.model = Mock()
        self.assertTrue(self.detection_service.is_model_loaded())
    
    def test_detect_cats_no_model(self):
        """Test detection when model is not loaded."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = self.detection_service.detect_cats(frame)
        self.assertEqual(len(detections), 0)


class TestProfileService(unittest.TestCase):
    """Test cases for ProfileService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_manager = ProfileManager(self.temp_dir)
        self.profile_service = ProfileService(self.profile_manager)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_profile_success(self):
        """Test successful profile creation."""
        profile_data = {
            'profile_name': 'test_cat',
            'display_name': 'Test Cat',
            'breed': 'Persian'
        }
        
        # Mock image paths
        image_paths = ['test1.jpg', 'test2.jpg']
        
        with patch.object(self.profile_service, '_create_color_signature') as mock_signature:
            mock_signature.return_value = np.array([100, 50, 200])
            
            with patch.object(self.profile_manager, 'save_profile') as mock_save:
                mock_save.return_value = True
                
                result = self.profile_service.create_profile(profile_data, image_paths)
                
                self.assertTrue(result)
                mock_signature.assert_called_once_with(image_paths)
                mock_save.assert_called_once()
    
    def test_create_profile_no_signature(self):
        """Test profile creation when color signature fails."""
        profile_data = {
            'profile_name': 'test_cat',
            'display_name': 'Test Cat',
            'breed': 'Persian'
        }
        
        image_paths = ['test1.jpg', 'test2.jpg']
        
        with patch.object(self.profile_service, '_create_color_signature') as mock_signature:
            mock_signature.return_value = None
            
            result = self.profile_service.create_profile(profile_data, image_paths)
            
            self.assertFalse(result)
    
    @patch('services.cv2.imread')
    @patch('services.extract_dominant_colors')
    def test_create_color_signature_success(self, mock_extract, mock_imread):
        """Test successful color signature creation."""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_extract.return_value = np.array([100, 50, 200])
        
        image_paths = ['test1.jpg', 'test2.jpg']
        signature = self.profile_service._create_color_signature(image_paths)
        
        self.assertIsNotNone(signature)
        self.assertEqual(len(signature), 3)
        self.assertEqual(mock_extract.call_count, 2)
    
    def test_create_color_signature_no_valid_images(self):
        """Test color signature creation with no valid images."""
        with patch('services.cv2.imread', return_value=None):
            image_paths = ['test1.jpg', 'test2.jpg']
            signature = self.profile_service._create_color_signature(image_paths)
            
            self.assertIsNone(signature)


class TestVideoRecordingService(unittest.TestCase):
    """Test cases for VideoRecordingService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recording_service = VideoRecordingService()
    
    def test_initial_state(self):
        """Test initial recording service state."""
        self.assertIsNone(self.recording_service.video_writer)
        self.assertIsNone(self.recording_service.current_recording)
        self.assertFalse(self.recording_service.is_recording)
        self.assertIsNone(self.recording_service.recording_start_time)
        self.assertIsNone(self.recording_service.last_seen_time)
    
    @patch('services.cv2.VideoWriter')
    def test_start_recording_success(self, mock_video_writer):
        """Test successful recording start."""
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.recording_service.start_recording("Test Cat", frame)
        
        self.assertTrue(result)
        self.assertTrue(self.recording_service.is_recording)
        self.assertIsNotNone(self.recording_service.current_recording)
        self.assertEqual(self.recording_service.current_recording.cat_name, "Test Cat")
    
    def test_start_recording_already_recording(self):
        """Test starting recording when already recording."""
        self.recording_service.is_recording = True
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.recording_service.start_recording("Test Cat", frame)
        
        self.assertFalse(result)
    
    def test_stop_recording_success(self):
        """Test successful recording stop."""
        mock_writer = Mock()
        self.recording_service.video_writer = mock_writer
        self.recording_service.is_recording = True
        
        result = self.recording_service.stop_recording()
        
        self.assertTrue(result)
        self.assertFalse(self.recording_service.is_recording)
        mock_writer.release.assert_called_once()
    
    def test_stop_recording_not_recording(self):
        """Test stopping recording when not recording."""
        result = self.recording_service.stop_recording()
        self.assertFalse(result)
    
    def test_update_last_seen(self):
        """Test last seen timestamp update."""
        self.recording_service.update_last_seen()
        self.assertIsNotNone(self.recording_service.last_seen_time)
    
    def test_should_stop_recording(self):
        """Test recording stop condition check."""
        import time
        
        # Not recording
        self.assertFalse(self.recording_service.should_stop_recording())
        
        # Recording but no last seen time
        self.recording_service.is_recording = True
        self.assertFalse(self.recording_service.should_stop_recording())
        
        # Recording with old last seen time
        self.recording_service.last_seen_time = time.time() - 10  # 10 seconds ago
        self.assertTrue(self.recording_service.should_stop_recording())


class TestApplicationService(unittest.TestCase):
    """Test cases for ApplicationService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.app_service = ApplicationService()
        # Override the profile manager to use temp directory
        self.app_service.profile_manager = ProfileManager(self.temp_dir)
        self.app_service.profile_service = ProfileService(self.app_service.profile_manager)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initial_state(self):
        """Test initial application state."""
        self.assertIsInstance(self.app_service.profile_manager, ProfileManager)
        self.assertIsInstance(self.app_service.detection_service, DetectionService)
        self.assertIsInstance(self.app_service.profile_service, ProfileService)
        self.assertIsInstance(self.app_service.recording_service, VideoRecordingService)
        self.assertIsInstance(self.app_service.state, ApplicationState)
    
    @patch.object(DetectionService, 'initialize_model')
    def test_initialize_success(self, mock_init):
        """Test successful application initialization."""
        mock_init.return_value = True
        
        result = self.app_service.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.app_service.state.model_loaded)
        mock_init.assert_called_once()
    
    @patch.object(DetectionService, 'initialize_model')
    def test_initialize_failure(self, mock_init):
        """Test application initialization failure."""
        mock_init.return_value = False
        
        result = self.app_service.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.app_service.state.model_loaded)
    
    def test_set_active_profile(self):
        """Test setting active profile."""
        # Test with None profile
        self.app_service.set_active_profile(None)
        self.assertIsNone(self.app_service.state.active_profile)
        
        # Test with "None" string
        self.app_service.set_active_profile("None (General Detection)")
        self.assertIsNone(self.app_service.state.active_profile)
        
        # Test with specific profile
        profile = CatProfile("test", "Test", "Persian")
        self.app_service.profile_manager.profiles["test"] = profile
        
        self.app_service.set_active_profile("test")
        self.assertEqual(self.app_service.state.active_profile, profile)
    
    def test_cleanup(self):
        """Test application cleanup."""
        self.app_service.recording_service.is_recording = True
        
        with patch.object(self.app_service.recording_service, 'stop_recording') as mock_stop:
            self.app_service.cleanup()
            mock_stop.assert_called_once()
            self.assertFalse(self.app_service.state.is_recording)


if __name__ == '__main__':
    unittest.main()
