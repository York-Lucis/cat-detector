"""
UI components for the Cat Detector application.
Contains all GUI-related classes and components.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Listbox
from PIL import Image, ImageTk
import threading
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

from models import CatProfile
from config import (
    WINDOW_TITLE, WINDOW_SIZE, PROFILE_MANAGER_SIZE, COLORS, STATUS_MESSAGES, ERROR_MESSAGES,
    MIN_PROFILE_IMAGES, MAX_PROFILE_IMAGES
)
from utils import validate_image_file, safe_filename
from logger import logger


class ProfileCreationDialog(simpledialog.Dialog):
    """Custom dialog for creating new cat profiles."""
    
    def __init__(self, parent, title: str = "Create New Cat Profile"):
        self.result = None
        super().__init__(parent, title)
    
    def body(self, master):
        """Create the dialog body."""
        # Profile name
        ttk.Label(master, text="Profile Name:").grid(row=0, sticky=tk.W, padx=5, pady=2)
        self.profile_name_entry = ttk.Entry(master, width=30)
        self.profile_name_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Display name
        ttk.Label(master, text="Display Name:").grid(row=1, sticky=tk.W, padx=5, pady=2)
        self.display_name_entry = ttk.Entry(master, width=30)
        self.display_name_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Breed
        ttk.Label(master, text="Breed/Race:").grid(row=2, sticky=tk.W, padx=5, pady=2)
        self.breed_entry = ttk.Entry(master, width=30)
        self.breed_entry.grid(row=2, column=1, padx=5, pady=2)
        
        return self.profile_name_entry  # Initial focus
    
    def apply(self):
        """Apply the dialog results."""
        self.result = {
            "profile_name": self.profile_name_entry.get().strip(),
            "display_name": self.display_name_entry.get().strip(),
            "breed": self.breed_entry.get().strip()
        }


class ProfileManager(tk.Toplevel):
    """Window for managing cat profiles."""
    
    def __init__(self, parent, profile_service, on_profile_change: Optional[Callable] = None):
        super().__init__(parent.root)
        self.parent = parent
        self.profile_service = profile_service
        self.on_profile_change = on_profile_change
        
        self.title("Profile Manager")
        self.geometry(PROFILE_MANAGER_SIZE)
        
        self.transient(parent.root)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self._create_widgets()
        self._populate_list()
    
    def _create_widgets(self):
        """Create the UI widgets."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Available Profiles:", font="-weight bold").pack(anchor=tk.W)
        
        # Profile list
        self.profile_listbox = Listbox(main_frame, height=10)
        self.profile_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.add_button = ttk.Button(btn_frame, text="Add New Profile", command=self._add_profile)
        self.add_button.pack(side=tk.LEFT, expand=True, padx=2)
        
        self.delete_button = ttk.Button(btn_frame, text="Delete Selected", command=self._delete_profile)
        self.delete_button.pack(side=tk.LEFT, expand=True, padx=2)
        
        # Status
        self.status_var = tk.StringVar(value=STATUS_MESSAGES['READY'])
        status_label = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, pady=(5, 0))
    
    def _populate_list(self):
        """Populate the profile list."""
        self.profile_listbox.delete(0, tk.END)
        try:
            profiles = self.profile_service.list_profiles()
            for profile in sorted(profiles):
                self.profile_listbox.insert(tk.END, profile)
        except Exception as e:
            logger.error(f"Error populating profile list: {e}")
            self.profile_listbox.insert(tk.END, "Error reading profiles")
    
    def _add_profile(self):
        """Add a new profile."""
        dialog = ProfileCreationDialog(self, "Create New Cat Profile")
        if not dialog.result:
            return
        
        profile_data = dialog.result
        profile_name = profile_data["profile_name"]
        
        # Validate input
        if not all(profile_data.values()):
            messagebox.showwarning("Incomplete Data", ERROR_MESSAGES['INCOMPLETE_DATA'], parent=self)
            return
        
        # Check if profile already exists
        if self.profile_service.get_profile(profile_name):
            messagebox.showerror("Error", ERROR_MESSAGES['PROFILE_EXISTS'].format(name=profile_name), parent=self)
            return
        
        # Select images
        filepaths = filedialog.askopenfilenames(
            title=f"Select {MIN_PROFILE_IMAGES}-{MAX_PROFILE_IMAGES} images for '{profile_name}'",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
            parent=self
        )
        
        if not filepaths:
            return
        
        # Validate image count
        if len(filepaths) < MIN_PROFILE_IMAGES or len(filepaths) > MAX_PROFILE_IMAGES:
            messagebox.showwarning(
                "Invalid Selection", 
                f"Please select between {MIN_PROFILE_IMAGES} and {MAX_PROFILE_IMAGES} images.",
                parent=self
            )
            return
        
        # Validate images
        valid_images = [path for path in filepaths if validate_image_file(path)]
        if len(valid_images) != len(filepaths):
            messagebox.showwarning("Invalid Images", "Some selected files are not valid images.", parent=self)
            return
        
        # Disable buttons and start processing
        self._set_processing_state(True)
        self.status_var.set(f"Status: Processing '{profile_name}'...")
        
        # Process in background thread
        thread = threading.Thread(
            target=self._process_profile_creation, 
            args=(profile_data, valid_images), 
            daemon=True
        )
        thread.start()
    
    def _process_profile_creation(self, profile_data: Dict[str, Any], image_paths: List[str]):
        """Process profile creation in background thread."""
        try:
            # Create safe profile name
            safe_name = safe_filename(profile_data['profile_name'])
            profile_data['profile_name'] = safe_name
            
            # Create profile
            success = self.profile_service.create_profile(profile_data, image_paths)
            
            # Update UI in main thread
            self.after(0, self._finish_profile_creation, success, profile_data['profile_name'])
            
        except Exception as e:
            logger.error(f"Profile creation thread error: {e}")
            self.after(0, self._finish_profile_creation, False, profile_data.get('profile_name', 'Unknown'))
    
    def _finish_profile_creation(self, success: bool, profile_name: str):
        """Finish profile creation and update UI."""
        if success:
            messagebox.showinfo("Success", ERROR_MESSAGES['PROFILE_CREATED'].format(name=profile_name), parent=self)
        else:
            messagebox.showerror("Error", ERROR_MESSAGES['PROFILE_CREATION_FAILED'].format(name=profile_name), parent=self)
        
        self._populate_list()
        self._set_processing_state(False)
        self.status_var.set(STATUS_MESSAGES['READY'])
        
        # Notify parent of profile change
        if self.on_profile_change:
            self.on_profile_change()
    
    def _delete_profile(self):
        """Delete selected profile."""
        selected_indices = self.profile_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", ERROR_MESSAGES['NO_SELECTION'], parent=self)
            return
        
        profile_name = self.profile_listbox.get(selected_indices[0])
        
        if messagebox.askyesno("Confirm Delete", ERROR_MESSAGES['CONFIRM_DELETE'].format(name=profile_name), parent=self):
            success = self.profile_service.delete_profile(profile_name)
            if success:
                self._populate_list()
                if self.on_profile_change:
                    self.on_profile_change()
            else:
                messagebox.showerror("Error", f"Failed to delete profile '{profile_name}'", parent=self)
    
    def _set_processing_state(self, processing: bool):
        """Set the processing state of buttons."""
        state = "disabled" if processing else "normal"
        self.add_button.config(state=state)
        self.delete_button.config(state=state)
    
    def on_closing(self):
        """Handle window closing."""
        if self.on_profile_change:
            self.on_profile_change()
        self.grab_release()
        self.destroy()


class VideoDisplay(ttk.Label):
    """Custom video display widget."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.imgtk = None
    
    def update_frame(self, frame):
        """Update the display with a new frame."""
        try:
            from PIL import Image
            import cv2
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            self.imgtk = ImageTk.PhotoImage(image=pil_image)
            self.config(image=self.imgtk, text="")
            
        except Exception as e:
            logger.error(f"Failed to update video display: {e}")
    
    def show_placeholder(self, text: str = "Webcam feed will appear here."):
        """Show placeholder text."""
        self.config(image='', text=text, background=COLORS['BACKGROUND'])


class StatusBar(ttk.Label):
    """Custom status bar widget."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, relief=tk.SUNKEN, anchor=tk.W, padding="5", **kwargs)
        self.status_var = tk.StringVar(value=STATUS_MESSAGES['INITIALIZING'])
        self.config(textvariable=self.status_var)
    
    def set_status(self, message: str):
        """Set the status message."""
        self.status_var.set(message)
    
    def set_detection_status(self, profile_name: str, device: str):
        """Set detection status."""
        message = STATUS_MESSAGES['DETECTING'].format(profile=profile_name, device=device.upper())
        self.set_status(message)
    
    def set_recording_status(self, duration: float):
        """Set recording status."""
        message = STATUS_MESSAGES['RECORDING'].format(duration=duration)
        self.set_status(message)


class ControlPanel(ttk.Frame):
    """Control panel with webcam and profile selection."""
    
    def __init__(self, parent, on_webcam_change: Optional[Callable] = None, 
                 on_profile_change: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.on_webcam_change = on_webcam_change
        self.on_profile_change = on_profile_change
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create control widgets."""
        # Webcam selection
        ttk.Label(self, text="Webcam:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.webcam_var = tk.StringVar()
        self.webcam_combo = ttk.Combobox(self, textvariable=self.webcam_var, state="readonly", width=15)
        self.webcam_combo.pack(side=tk.LEFT, padx=5)
        self.webcam_combo.bind('<<ComboboxSelected>>', self._on_webcam_change)
        
        # Profile selection
        ttk.Label(self, text="Profile:").pack(side=tk.LEFT, padx=(10, 5))
        
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(self, textvariable=self.profile_var, state="readonly", width=20)
        self.profile_combo.pack(side=tk.LEFT, padx=5)
        self.profile_combo.bind('<<ComboboxSelected>>', self._on_profile_change)
        
        # Buttons
        self.toggle_button = ttk.Button(self, text="Start Webcam", command=self._on_toggle_webcam)
        self.toggle_button.pack(side=tk.LEFT, padx=10)
        
        self.profile_button = ttk.Button(self, text="Manage Profiles...", command=self._on_manage_profiles)
        self.profile_button.pack(side=tk.LEFT, padx=5)
    
    def _on_webcam_change(self, event=None):
        """Handle webcam selection change."""
        if self.on_webcam_change:
            self.on_webcam_change(self.webcam_var.get())
    
    def _on_profile_change(self, event=None):
        """Handle profile selection change."""
        if self.on_profile_change:
            self.on_profile_change(self.profile_var.get())
    
    def _on_toggle_webcam(self):
        """Handle webcam toggle button."""
        # This will be connected to the main application
        pass
    
    def _on_manage_profiles(self):
        """Handle profile management button."""
        # This will be connected to the main application
        pass
    
    def populate_webcams(self, webcams: List[int]):
        """Populate webcam dropdown."""
        if not webcams:
            self.webcam_combo['values'] = [ERROR_MESSAGES['NO_WEBCAMS']]
            self.webcam_var.set(ERROR_MESSAGES['NO_WEBCAMS'])
        else:
            webcam_names = [f"Webcam {i}" for i in webcams]
            self.webcam_combo['values'] = webcam_names
            self.webcam_var.set(webcam_names[0])
    
    def populate_profiles(self, profiles: List[str]):
        """Populate profile dropdown."""
        display_names = ["None (General Detection)"] + sorted(profiles)
        self.profile_combo['values'] = display_names
        self.profile_var.set(display_names[0])
    
    def set_webcam_toggle_callback(self, callback: Callable):
        """Set the webcam toggle callback."""
        self.toggle_button.config(command=callback)
    
    def set_profile_manage_callback(self, callback: Callable):
        """Set the profile management callback."""
        self.profile_button.config(command=callback)
    
    def get_selected_webcam(self) -> Optional[int]:
        """Get the selected webcam index."""
        try:
            webcam_text = self.webcam_var.get()
            if webcam_text == ERROR_MESSAGES['NO_WEBCAMS']:
                return None
            return int(webcam_text.split()[-1])
        except (ValueError, IndexError):
            return None
    
    def get_selected_profile(self) -> str:
        """Get the selected profile name."""
        return self.profile_var.get()
    
    def set_webcam_button_text(self, text: str):
        """Set the webcam toggle button text."""
        self.toggle_button.config(text=text)
