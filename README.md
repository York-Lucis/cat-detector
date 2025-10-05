# 🐱 Cat Detector & Recognition System

A real-time cat detection and recognition application that uses computer vision and machine learning to identify specific cats through their unique color profiles. Built with YOLOv8 for object detection and custom color analysis for individual cat recognition.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Real-time Cat Detection**: Uses YOLOv8 model to detect cats in live webcam feed
- **Individual Cat Recognition**: Creates unique color profiles for specific cats using HSV color analysis
- **Profile Management**: Easy-to-use GUI for adding, managing, and deleting cat profiles
- **Multi-Webcam Support**: Automatically detects and supports multiple webcam devices
- **GPU Acceleration**: Supports both CPU and CUDA GPU processing for optimal performance
- **Intuitive GUI**: Clean, user-friendly interface built with Tkinter

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- (Optional) CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/York-Lucis/cat-detector.git
   cd cat-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## 📖 How It Works

### Detection Pipeline

1. **Object Detection**: YOLOv8 model identifies cats in the video stream
2. **Color Analysis**: For each detected cat, the system extracts the region of interest (ROI)
3. **HSV Conversion**: Converts the cat's image to HSV color space for better color analysis
4. **Profile Matching**: Compares the cat's average color with stored profiles using Euclidean distance
5. **Recognition**: If a match is found within tolerance, displays the cat's name

### Profile Creation

The system creates unique profiles for individual cats by:

1. **Image Collection**: Users provide 2-4 clear images of the cat
2. **Color Clustering**: Uses K-Means clustering to identify dominant colors in each image
3. **Profile Generation**: Calculates the average dominant color across all images
4. **Storage**: Saves the color signature as a NumPy array and metadata as JSON

## 🎯 Usage

### Creating Cat Profiles

1. Click **"Manage Profiles..."** button
2. Click **"Add New Profile"**
3. Fill in the cat's information:
   - Profile Name (unique identifier)
   - Display Name (shown during recognition)
   - Breed/Race (optional)
4. Select 2-4 clear images of the cat
5. Wait for profile processing to complete

### Running Detection

1. Select your webcam from the dropdown
2. Choose a cat profile (or "None" for general detection)
3. Click **"Start Webcam"** to begin detection
4. The system will highlight detected cats:
   - **Green box**: General cat detection
   - **Red box**: Recognized specific cat

## 🛠️ Technical Details

### Architecture

- **Frontend**: Tkinter-based GUI with real-time video display
- **Backend**: YOLOv8 for object detection, OpenCV for image processing
- **Color Analysis**: K-Means clustering and HSV color space analysis
- **Threading**: Background processing for profile creation and model loading

### Key Components

- `CatDetectorApp`: Main application class handling GUI and detection logic
- `ProfileManager`: Manages cat profile creation, deletion, and storage
- `ProfileCreationDialog`: Custom dialog for profile information input
- Color matching algorithm with configurable tolerance

### Configuration

```python
PROFILES_FOLDER = "cat_profiles"          # Storage directory for profiles
COLOR_MATCH_TOLERANCE = 35                # Color matching sensitivity
PROFILE_FILENAME = "profile.npy"          # Color signature file
PROFILE_METADATA_FILENAME = "profile.json" # Profile metadata file
```

## 📁 Project Structure

```
cat-detector/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── yolov8n.pt            # YOLOv8 model weights
├── cat_profiles/         # Cat profile storage
│   └── [profile_name]/
│       ├── profile.npy   # Color signature
│       ├── profile.json  # Metadata
│       └── *.jpg         # Profile images
└── README.md             # This file
```

## 🔧 Dependencies

- **opencv-python**: Computer vision and image processing
- **ultralytics**: YOLOv8 model implementation
- **numpy**: Numerical computations and array operations
- **scikit-learn**: K-Means clustering for color analysis
- **Pillow**: Image handling and GUI integration
- **torch**: PyTorch for model inference
- **torchvision**: Computer vision utilities

## 🎨 Customization

### Adjusting Recognition Sensitivity

Modify the `COLOR_MATCH_TOLERANCE` value in `main.py`:
- **Lower values** (e.g., 25): More strict matching, fewer false positives
- **Higher values** (e.g., 45): More lenient matching, may include false positives

### Adding New Features

The modular design allows for easy extension:
- Additional color spaces for analysis
- Face recognition integration
- Multiple cat detection in single frame
- Export/import profile functionality

## 🐛 Troubleshooting

### Common Issues

1. **"No Webcams Found"**
   - Ensure your webcam is connected and not used by other applications
   - Try different camera indices in the code

2. **Model Loading Errors**
   - Check internet connection for YOLOv8 model download
   - Verify PyTorch installation

3. **Poor Recognition Accuracy**
   - Use higher quality, well-lit images for profiles
   - Ensure consistent lighting conditions
   - Adjust `COLOR_MATCH_TOLERANCE` value

4. **Performance Issues**
   - Use GPU acceleration if available
   - Reduce webcam resolution
   - Close other resource-intensive applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the [Issues](https://github.com/York-Lucis/cat-detector/issues) page
2. Create a new issue with detailed information
3. Include system specifications and error messages

---

**Made with ❤️ for cat lovers and computer vision enthusiasts**
