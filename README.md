# Sound Detection App

A Python application for recording audio, training sound recognition models, and real-time sound detection.

## Features

1. **Audio Recording**: Record 5-second audio clips and save them to files
2. **Model Training**: Train machine learning models to recognize specific sounds
3. **Real-time Detection**: Continuously monitor microphone input for trained sounds

## Installation

### Prerequisites

- Python 3.8 or higher
- Microphone access
- Sufficient disk space for audio recordings and models

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd sound-detect
   ```

2. **Option 1: Automatic Installation (Recommended)**
   ```bash
   python install_dependencies.py
   ```

3. **Option 2: Manual Installation**
   
   **For Windows:**
   ```bash
   # Install pipwin for PyAudio
   pip install pipwin
   pipwin install pyaudio
   
   # Install other dependencies
   pip install numpy pandas scikit-learn matplotlib librosa soundfile tensorflow
   ```
   
   **For Linux/macOS:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Using Conda:**
   ```bash
   conda install numpy pandas scikit-learn matplotlib librosa soundfile tensorflow
   conda install -c conda-forge pyaudio
   ```

4. Verify installation:
   ```bash
   python test_recording.py
   ```

## Usage

### Quick Start

1. **Test basic functionality**:
   ```bash
   python test_recording.py
   ```

2. **Run comprehensive test**:
   ```bash
   python test_all_features.py
   ```

3. **Run the full application**:
   ```bash
   python main.py
   ```

### Using the Application

The application has three main tabs:

#### 1. Recording Tab
- Enter a sound type (e.g., "doorbell", "alarm", "voice")
- Click "Record 5 Seconds" to capture audio
- View and manage your recordings
- Play back recordings to verify quality

#### 2. Training Tab
- Train a model using your recorded sounds
- Load existing trained models
- Test model performance

#### 3. Detection Tab
- Start real-time sound detection
- Monitor for specific sounds
- View detection results

## Project Structure

```
sound-detect/
├── src/
│   ├── __init__.py
│   ├── audio_utils.py      # Core audio processing functions
│   ├── audio_recorder.py   # Audio recording functionality
│   ├── model_trainer.py    # Model training functionality
│   ├── sound_detector.py   # Real-time detection
│   └── gui.py             # User interface
├── data/
│   ├── recordings/        # Raw audio recordings
│   ├── processed/         # Processed audio features
│   └── models/           # Trained models
├── main.py               # Main application entry point
├── test_recording.py     # Basic functionality test
├── test_all_features.py  # Comprehensive feature test
├── install_dependencies.py # Dependency installer
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **PyAudio installation fails**:
   - **Windows**: Use `pip install pipwin && pipwin install pyaudio`
   - **Linux**: Install portaudio first: `sudo apt-get install portaudio19-dev`
   - **macOS**: Use Homebrew: `brew install portaudio`

2. **Numpy installation fails**:
   - Try: `pip install numpy --upgrade`
   - Or use conda: `conda install numpy`

3. **Microphone not detected**:
   - Check system microphone permissions
   - Ensure microphone is not being used by another application
   - Test with system audio settings

4. **Import errors**:
   - Run the automatic installer: `python install_dependencies.py`
   - Check Python version (3.8+ required)
   - Try installing packages individually

5. **Recording fails**:
   - Check microphone permissions
   - Ensure microphone is working in other applications
   - Try running `python test_recording.py` first

### Windows-Specific Solutions

If you're having issues on Windows:

1. **Use the automatic installer**:
   ```bash
   python install_dependencies.py
   ```

2. **Manual PyAudio installation**:
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

3. **Alternative PyAudio installation**:
   - Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Install with: `pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl`

4. **Use Conda instead of pip**:
   ```bash
   conda install -c conda-forge pyaudio
   conda install numpy pandas scikit-learn matplotlib librosa soundfile tensorflow
   ```

### Getting Help

If you encounter issues:

1. Run the automatic installer: `python install_dependencies.py`
2. Run the test script: `python test_recording.py`
3. Check the console output for error messages
4. Ensure all dependencies are properly installed
5. Verify microphone access and permissions

## Development

### Adding New Features

The application is modular and can be extended:

- **New audio formats**: Modify `audio_utils.py`
- **Different models**: Add new training algorithms
- **Enhanced GUI**: Extend `gui.py` with new features

### Testing

Run the test script to verify functionality:
```bash
python test_recording.py
```

## License

This project is open source. Feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request 