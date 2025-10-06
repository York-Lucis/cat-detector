"""
Data models for the Cat Detector application.

This module defines the core data structures used throughout the application,
including cat profiles, detection results, application state, and profile management.
All models are designed to be immutable where possible and provide clear interfaces
for data manipulation and persistence.

Author: Cat Detector Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CatProfile:
    """
    Represents a cat profile with metadata and color signature.
    
    This class encapsulates all information about a specific cat, including
    its unique identifier, display information, and color signature used for
    recognition. The color signature is a numpy array representing the average
    HSV color values extracted from the cat's profile images.
    
    Attributes:
        profile_name: Unique identifier for the profile (used for file storage)
        display_name: Human-readable name shown in the UI
        breed: Cat breed or race information
        signature: HSV color signature array for recognition (3-element array)
        image_paths: List of image file paths used to create the profile
        created_at: Timestamp when the profile was created
        updated_at: Timestamp when the profile was last modified
    """
    
    profile_name: str
    display_name: str
    breed: str
    signature: Optional[np.ndarray] = None
    image_paths: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """
        Initialize timestamps if not provided.
        
        This method is called automatically after object initialization
        to set default timestamps if they weren't provided during creation.
        """
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the profile suitable for JSON serialization.
            Note that the signature is not included as it's stored separately.
        """
        return {
            'profile_name': self.profile_name,
            'display_name': self.display_name,
            'breed': self.breed,
            'image_paths': self.image_paths,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CatProfile':
        """
        Create profile from dictionary.
        
        Args:
            data: Dictionary containing profile data
            
        Returns:
            CatProfile instance created from the dictionary data
            
        Raises:
            KeyError: If required fields are missing from the data
            ValueError: If timestamp data is in invalid format
        """
        profile = cls(
            profile_name=data['profile_name'],
            display_name=data['display_name'],
            breed=data['breed'],
            image_paths=data.get('image_paths', [])
        )
        
        if data.get('created_at'):
            profile.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            profile.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return profile
    
    def update_timestamp(self) -> None:
        """
        Update the updated_at timestamp.
        
        This method should be called whenever the profile is modified
        to maintain accurate tracking of when changes were made.
        """
        self.updated_at = datetime.now()


@dataclass
class DetectionResult:
    """
    Represents the result of a cat detection.
    
    This class encapsulates information about a detected cat, including its
    location, confidence level, and whether it matches a specific profile.
    
    Attributes:
        confidence: Detection confidence score (0.0 to 1.0)
        bounding_box: Tuple of (x1, y1, x2, y2) coordinates defining the detection box
        is_specific_cat: Whether this detection matches a specific cat profile
        cat_name: Name of the recognized cat (if is_specific_cat is True)
        color_distance: Color distance from the matched profile (if applicable)
    """
    
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_specific_cat: bool = False
    cat_name: Optional[str] = None
    color_distance: Optional[float] = None
    
    @property
    def x1(self) -> int:
        return int(self.bounding_box[0])
    
    @property
    def y1(self) -> int:
        return int(self.bounding_box[1])
    
    @property
    def x2(self) -> int:
        return int(self.bounding_box[2])
    
    @property
    def y2(self) -> int:
        return int(self.bounding_box[3])
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class VideoRecording:
    """Represents an active video recording session."""
    
    filename: str
    start_time: datetime
    cat_name: str
    is_active: bool = True
    duration: float = 0.0
    
    def update_duration(self, current_time: datetime):
        """Update the recording duration."""
        self.duration = (current_time - self.start_time).total_seconds()


@dataclass
class ApplicationState:
    """Represents the current state of the application."""
    
    is_running: bool = False
    is_recording: bool = False
    active_profile: Optional[CatProfile] = None
    selected_webcam: Optional[int] = None
    device: str = 'cpu'
    model_loaded: bool = False
    last_detection_time: Optional[datetime] = None
    current_recording: Optional[VideoRecording] = None
    
    def reset_recording_state(self):
        """Reset recording-related state."""
        self.is_recording = False
        self.current_recording = None
        self.last_detection_time = None


@dataclass
class WebcamInfo:
    """Information about an available webcam."""
    
    index: int
    name: str
    is_available: bool = True
    
    def __str__(self) -> str:
        return f"Webcam {self.index}"


class ProfileManager:
    """Manages cat profiles and their persistence."""
    
    def __init__(self, profiles_folder: Path):
        self.profiles_folder = Path(profiles_folder)
        self.profiles: Dict[str, CatProfile] = {}
        self._ensure_profiles_folder()
    
    def _ensure_profiles_folder(self):
        """Ensure the profiles folder exists."""
        self.profiles_folder.mkdir(exist_ok=True)
    
    def load_profiles(self) -> Dict[str, CatProfile]:
        """Load all profiles from the filesystem."""
        self.profiles.clear()
        
        if not self.profiles_folder.exists():
            return self.profiles
        
        for profile_dir in self.profiles_folder.iterdir():
            if not profile_dir.is_dir():
                continue
                
            try:
                profile = self._load_single_profile(profile_dir)
                if profile:
                    self.profiles[profile.profile_name] = profile
            except Exception as e:
                print(f"Error loading profile from {profile_dir}: {e}")
        
        return self.profiles
    
    def _load_single_profile(self, profile_dir: Path) -> Optional[CatProfile]:
        """Load a single profile from a directory."""
        metadata_file = profile_dir / "profile.json"
        signature_file = profile_dir / "profile.npy"
        
        if not metadata_file.exists() or not signature_file.exists():
            return None
        
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load color signature
        signature = np.load(signature_file)
        
        # Create profile
        profile = CatProfile.from_dict(metadata)
        profile.signature = signature
        
        # Get image paths
        image_extensions = {'.jpg', '.jpeg', '.png'}
        profile.image_paths = [
            str(img_file) for img_file in profile_dir.iterdir()
            if img_file.suffix.lower() in image_extensions
        ]
        
        return profile
    
    def save_profile(self, profile: CatProfile) -> bool:
        """Save a profile to the filesystem."""
        try:
            profile_dir = self.profiles_folder / profile.profile_name
            profile_dir.mkdir(exist_ok=True)
            
            # Save metadata
            metadata_file = profile_dir / "profile.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, indent=4, ensure_ascii=False)
            
            # Save signature
            if profile.signature is not None:
                signature_file = profile_dir / "profile.npy"
                np.save(signature_file, profile.signature)
            
            # Update in-memory profiles
            self.profiles[profile.profile_name] = profile
            
            return True
        except Exception as e:
            print(f"Error saving profile {profile.profile_name}: {e}")
            return False
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile from the filesystem."""
        try:
            profile_dir = self.profiles_folder / profile_name
            if profile_dir.exists():
                import shutil
                shutil.rmtree(profile_dir)
            
            # Remove from in-memory profiles
            self.profiles.pop(profile_name, None)
            
            return True
        except Exception as e:
            print(f"Error deleting profile {profile_name}: {e}")
            return False
    
    def get_profile(self, profile_name: str) -> Optional[CatProfile]:
        """Get a profile by name."""
        return self.profiles.get(profile_name)
    
    def list_profiles(self) -> List[str]:
        """Get a list of all profile names."""
        return list(self.profiles.keys())
    
    def profile_exists(self, profile_name: str) -> bool:
        """Check if a profile exists."""
        return profile_name in self.profiles
