#!/usr/bin/env python3
"""
Main entry point for the Sound Detection App.
Launches the GUI and handles application initialization.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.gui import SoundDetectionGUI
    from src.audio_recorder import RecordingManager
    from src.audio_utils import AudioRecorder
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import pyaudio
    except ImportError:
        missing_deps.append("pyaudio")
        
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
        
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import soundfile
    except ImportError:
        missing_deps.append("soundfile")
        
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install -r requirements.txt")
        return False
        
    return True


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data",
        "data/recordings",
        "data/processed", 
        "data/models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    """Main application function."""
    print("Starting Sound Detection App...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Create directories
    create_directories()
    
    # Create main window
    root = tk.Tk()
    
    try:
        # Initialize GUI
        app = SoundDetectionGUI(root)
        
        # Handle window close
        def on_closing():
            try:
                app.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
            root.destroy()
            
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start GUI
        print("GUI initialized successfully")
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {e}")
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 