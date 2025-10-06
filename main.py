import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Listbox
from PIL import Image, ImageTk
import shutil
import threading
import torch
import json
import time
from datetime import datetime

# --- Configuration ---
PROFILES_FOLDER = "cat_profiles"
COLOR_MATCH_TOLERANCE = 35  # Slightly increased tolerance
PROFILE_FILENAME = "profile.npy"
PROFILE_METADATA_FILENAME = "profile.json"
VIDEO_RECORDINGS_FOLDER = "video_recordings"
MAX_RECORDING_DURATION = 60  # Maximum recording duration in seconds

class ProfileCreationDialog(simpledialog.Dialog):
    """A custom dialog to get detailed info for a new cat profile."""
    def body(self, master):
        ttk.Label(master, text="Profile Name:").grid(row=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(master, text="Display Name:").grid(row=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(master, text="Breed/Race:").grid(row=2, sticky=tk.W, padx=5, pady=2)

        self.profile_name_entry = ttk.Entry(master, width=30)
        self.display_name_entry = ttk.Entry(master, width=30)
        self.breed_entry = ttk.Entry(master, width=30)

        self.profile_name_entry.grid(row=0, column=1, padx=5, pady=2)
        self.display_name_entry.grid(row=1, column=1, padx=5, pady=2)
        self.breed_entry.grid(row=2, column=1, padx=5, pady=2)
        
        return self.profile_name_entry # initial focus

    def apply(self):
        self.result = {
            "profile_name": self.profile_name_entry.get().strip(),
            "display_name": self.display_name_entry.get().strip(),
            "breed": self.breed_entry.get().strip()
        }

class ProfileManager(tk.Toplevel):
    """A Toplevel window for adding, removing, and viewing cat profiles."""
    def __init__(self, parent):
        super().__init__(parent.root)
        self.parent = parent
        self.title("Profile Manager")
        self.geometry("400x350")
        
        self.transient(parent.root)
        self.grab_set()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Available Profiles:", font="-weight bold").pack(anchor=tk.W)

        self.profile_listbox = Listbox(main_frame, height=10)
        self.profile_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.add_button = ttk.Button(btn_frame, text="Add New Profile", command=self.add_profile)
        self.add_button.pack(side=tk.LEFT, expand=True, padx=2)
        self.delete_button = ttk.Button(btn_frame, text="Delete Selected", command=self.delete_profile)
        self.delete_button.pack(side=tk.LEFT, expand=True, padx=2)
        
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, pady=(5,0))

        self.populate_list()

    def populate_list(self):
        self.profile_listbox.delete(0, tk.END)
        profiles = [d for d in os.listdir(PROFILES_FOLDER) if os.path.isdir(os.path.join(PROFILES_FOLDER, d))]
        for profile in sorted(profiles):
            self.profile_listbox.insert(tk.END, profile)

    def add_profile(self):
        dialog = ProfileCreationDialog(self, "Create New Cat Profile")
        if not dialog.result: return

        profile_data = dialog.result
        profile_name = profile_data["profile_name"]

        if not all(profile_data.values()):
            messagebox.showwarning("Incomplete Data", "All fields are required.", parent=self)
            return

        profile_path = os.path.join(PROFILES_FOLDER, profile_name)
        if os.path.exists(profile_path):
            messagebox.showerror("Error", f"A profile named '{profile_name}' already exists.", parent=self)
            return

        filepaths = filedialog.askopenfilenames(
            title=f"Select 2-4 images for '{profile_name}'",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
            parent=self
        )
        if not filepaths: return

        self.add_button.config(state="disabled")
        self.delete_button.config(state="disabled")
        self.status_var.set(f"Status: Processing '{profile_name}'...")
        
        thread = threading.Thread(target=self._process_profile_creation, args=(profile_data, profile_path, filepaths), daemon=True)
        thread.start()

    def _process_profile_creation(self, profile_data, profile_path, filepaths):
        """Worker function to be run in a background thread."""
        os.makedirs(profile_path, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(profile_path, PROFILE_METADATA_FILENAME), 'w') as f:
            json.dump(profile_data, f, indent=4)

        for path in filepaths:
            shutil.copy(path, profile_path)

        success = self.create_color_profile_file(profile_path)
        self.after(0, self._finish_profile_creation, success, profile_data['profile_name'], profile_path)

    def _finish_profile_creation(self, success, profile_name, profile_path):
        """Updates the GUI after background processing is complete."""
        if success:
            messagebox.showinfo("Success", f"Profile '{profile_name}' was created successfully.", parent=self)
        else:
            messagebox.showerror("Error", f"Could not create a color profile for '{profile_name}'. Please use clearer images.", parent=self)
            if os.path.exists(profile_path): shutil.rmtree(profile_path)

        self.populate_list()
        self.status_var.set("Status: Ready")
        self.add_button.config(state="normal")
        self.delete_button.config(state="normal")

    def create_color_profile_file(self, profile_path):
        """Analyzes images and saves a dominant color profile file."""
        image_files = [f for f in os.listdir(profile_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_dominant_colors = []
        for filename in image_files:
            img = cv2.imread(os.path.join(profile_path, filename))
            if img is None: continue
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pixels = img_hsv.reshape(-1, 3)
            try:
                kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0).fit(pixels)
                counts = np.bincount(kmeans.labels_)
                dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
                all_dominant_colors.append(dominant_color)
            except Exception as e:
                print(f"KMeans failed for {filename}: {e}")
                continue
        if not all_dominant_colors: return False
        avg_profile = np.mean(all_dominant_colors, axis=0)
        np.save(os.path.join(profile_path, PROFILE_FILENAME), avg_profile)
        return True

    def delete_profile(self):
        selected_indices = self.profile_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a profile to delete.", parent=self)
            return
        profile_name = self.profile_listbox.get(selected_indices[0])
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the profile '{profile_name}'?", parent=self):
            shutil.rmtree(os.path.join(PROFILES_FOLDER, profile_name))
            self.populate_list()
    
    def on_closing(self):
        self.parent.load_and_populate_profiles()
        self.grab_release()
        self.destroy()

class CatDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Recognition Tool")
        self.root.geometry("1000x750")

        self.cap = None
        self.is_running = False
        self.active_profile = None
        self.model = None
        self.device = None
        self.profiles = {}
        
        # Video recording variables
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.recording_cat_name = None
        self.recording_fps = 30
        self.recording_width = 640
        self.recording_height = 480

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(main_frame, text="Webcam feed will appear here.", anchor=tk.CENTER, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=10)

        controls_frame = ttk.Frame(main_frame, padding="10")
        controls_frame.pack(fill=tk.X)
        
        ttk.Label(controls_frame, text="Webcam:").pack(side=tk.LEFT, padx=(0, 5))
        self.webcam_var = tk.StringVar()
        self.webcam_combo = ttk.Combobox(controls_frame, textvariable=self.webcam_var, state="readonly", width=15)
        self.webcam_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(controls_frame, text="Profile:").pack(side=tk.LEFT, padx=(10, 5))
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(controls_frame, textvariable=self.profile_var, state="readonly", width=20)
        self.profile_combo.pack(side=tk.LEFT, padx=5)

        self.toggle_button = ttk.Button(controls_frame, text="Start Webcam", command=self.toggle_webcam)
        self.toggle_button.pack(side=tk.LEFT, padx=10)
        
        self.profile_button = ttk.Button(controls_frame, text="Manage Profiles...", command=self.open_profile_manager)
        self.profile_button.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Status: Initializing...")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        os.makedirs(PROFILES_FOLDER, exist_ok=True)
        os.makedirs(VIDEO_RECORDINGS_FOLDER, exist_ok=True)
        self.populate_webcams()
        
        threading.Thread(target=self.initialize_backend, daemon=True).start()

    def initialize_backend(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status_var.set(f"Status: Loading YOLOv8 model onto {self.device.upper()}...")
        try:
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            self.status_var.set(f"Status: Model loaded on {self.device.upper()}. Ready.")
        except Exception as e:
            self.status_var.set("Status: Error loading model. Check internet.")
            messagebox.showerror("Model Load Error", f"Failed to load YOLOv8 model: {e}")
            return
        
        self.load_and_populate_profiles()

    def load_and_populate_profiles(self):
        """Scans profile folders, loads .npy and .json files, and populates the dropdown."""
        self.profiles.clear()
        profile_names = [d for d in os.listdir(PROFILES_FOLDER) if os.path.isdir(os.path.join(PROFILES_FOLDER, d))]
        
        for name in profile_names:
            profile_file = os.path.join(PROFILES_FOLDER, name, PROFILE_FILENAME)
            metadata_file = os.path.join(PROFILES_FOLDER, name, PROFILE_METADATA_FILENAME)

            if os.path.exists(profile_file) and os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata['signature'] = np.load(profile_file)
                    self.profiles[name] = metadata
                except Exception as e:
                    print(f"Could not load profile '{name}': {e}")

        display_names = ["None (General Detection)"] + sorted(self.profiles.keys())
        self.profile_combo['values'] = display_names
        self.profile_var.set(display_names[0])

    def populate_webcams(self):
        available_cameras = [f"Webcam {i}" for i in range(10) if cv2.VideoCapture(i, cv2.CAP_DSHOW).isOpened()]
        if not available_cameras:
            self.webcam_combo['values'] = ["No Webcams Found"]; self.webcam_var.set("No Webcams Found")
        else:
            self.webcam_combo['values'] = available_cameras; self.webcam_var.set(available_cameras[0])

    def open_profile_manager(self):
        self.profile_button.config(state="disabled")
        manager = ProfileManager(self)
        self.root.wait_window(manager)
        self.profile_button.config(state="normal")

    def start_video_recording(self, cat_name):
        """Start recording video when a specific cat is detected."""
        if self.is_recording:
            return  # Already recording
        
        try:
            # Create timestamped folder for this recording session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_folder = os.path.join(VIDEO_RECORDINGS_FOLDER, f"{cat_name}_{timestamp}")
            os.makedirs(session_folder, exist_ok=True)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_filename = os.path.join(session_folder, f"{cat_name}_recording.mp4")
            self.video_writer = cv2.VideoWriter(
                video_filename, 
                fourcc, 
                self.recording_fps, 
                (self.recording_width, self.recording_height)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Could not initialize video writer")
            
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_cat_name = cat_name
            
            self.status_var.set(f"Status: Recording {cat_name}... | Duration: 0s")
            print(f"Started recording {cat_name} at {timestamp}")
            
        except Exception as e:
            print(f"Error starting video recording: {e}")
            self.status_var.set(f"Status: Recording error - {str(e)}")

    def stop_video_recording(self):
        """Stop recording and save the video."""
        if not self.is_recording or not self.video_writer:
            return
        
        try:
            self.video_writer.release()
            self.video_writer = None
            
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            cat_name = self.recording_cat_name or "Unknown"
            
            self.status_var.set(f"Status: Saved {cat_name} recording ({duration:.1f}s)")
            print(f"Stopped recording {cat_name}, duration: {duration:.1f}s")
            
        except Exception as e:
            print(f"Error stopping video recording: {e}")
            self.status_var.set(f"Status: Error saving recording - {str(e)}")
        finally:
            self.is_recording = False
            self.recording_start_time = None
            self.recording_cat_name = None

    def write_frame_to_video(self, frame):
        """Write a frame to the current video recording."""
        if self.is_recording and self.video_writer:
            try:
                # Resize frame to recording dimensions
                resized_frame = cv2.resize(frame, (self.recording_width, self.recording_height))
                self.video_writer.write(resized_frame)
                
                # Check if maximum recording duration reached
                if self.recording_start_time:
                    elapsed_time = time.time() - self.recording_start_time
                    if elapsed_time >= MAX_RECORDING_DURATION:
                        self.stop_video_recording()
                    else:
                        # Update status with recording duration
                        self.status_var.set(f"Status: Recording {self.recording_cat_name}... | Duration: {elapsed_time:.1f}s")
                        
            except Exception as e:
                print(f"Error writing frame to video: {e}")
                self.stop_video_recording()

    def toggle_webcam(self):
        if self.is_running:
            self.is_running = False
            self.toggle_button.config(text="Start Webcam")
            if self.cap: self.cap.release()
            # Stop any ongoing recording when webcam is stopped
            if self.is_recording:
                self.stop_video_recording()
            self.video_label.config(image='', background='black', text="Webcam feed stopped.")
        else:
            if not self.model: return messagebox.showerror("Error", "Model is not loaded yet.")
            
            selected_profile_name = self.profile_var.get()
            if selected_profile_name == "None (General Detection)": self.active_profile = None
            else: self.active_profile = self.profiles.get(selected_profile_name)

            try:
                cam_index = int(self.webcam_var.get().split()[-1])
                self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened(): raise ValueError("Cannot open camera.")
                self.is_running = True
                self.toggle_button.config(text="Stop Webcam")
                self.video_loop()
            except (ValueError, IndexError): messagebox.showerror("Webcam Error", f"Could not open {self.webcam_var.get()}.")

    def video_loop(self):
        if not self.is_running: return
        
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, verbose=False, device=self.device)
            profile_name = self.profile_var.get()
            
            # Check if we're currently recording and update status accordingly
            if not self.is_recording:
                self.status_var.set(f"Status: Detecting... | Profile: {profile_name} | Device: {self.device.upper()}")

            cat_detected = False
            specific_cat_detected = False
            detected_cat_name = None

            for result in results:
                cat_class_id = 15 # YOLOv8 COCO class for 'cat'
                for box in result.boxes:
                    if int(box.cls[0]) == cat_class_id:
                        cat_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        box_color, text, thickness = ((0, 255, 0), f"Cat: {float(box.conf[0]):.2f}", 2)

                        if self.active_profile and 'signature' in self.active_profile:
                            cat_roi = frame[y1:y2, x1:x2]
                            if cat_roi.size > 0:
                                cat_roi_hsv = cv2.cvtColor(cat_roi, cv2.COLOR_BGR2HSV)
                                avg_color = np.mean(cat_roi_hsv.reshape(-1, 3), axis=0)
                                color_dist = np.linalg.norm(avg_color - self.active_profile['signature'])
                                
                                if color_dist < COLOR_MATCH_TOLERANCE:
                                    specific_cat_detected = True
                                    display_name = self.active_profile.get('display_name', 'Match')
                                    detected_cat_name = display_name
                                    box_color, text, thickness = ((0, 0, 255), f"!!! {display_name} !!!", 3)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            # Handle video recording logic
            if specific_cat_detected and detected_cat_name:
                # Start recording if not already recording
                if not self.is_recording:
                    self.start_video_recording(detected_cat_name)
            elif not cat_detected and self.is_recording:
                # Stop recording if no cats detected and we were recording
                self.stop_video_recording()
            
            # Write frame to video if recording
            if self.is_recording:
                self.write_frame_to_video(frame)
            
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.imgtk = img_tk
            self.video_label.config(image=img_tk, text="")
        
        self.root.after(10, self.video_loop)

    def on_closing(self):
        self.is_running = False
        if self.cap: self.cap.release()
        # Stop any ongoing recording when application is closed
        if self.is_recording:
            self.stop_video_recording()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDetectorApp(root)
    root.mainloop()