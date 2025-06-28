#!/usr/bin/env python3
"""
Dependency installation script for Sound Detection App.
Handles Windows-specific installation issues and provides alternative methods.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_with_pip():
    """Install dependencies using pip."""
    print("\n" + "="*50)
    print("INSTALLING DEPENDENCIES WITH PIP")
    print("="*50)
    
    # Upgrade pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install numpy first (often causes issues)
    print("\nInstalling numpy first...")
    if not run_command("pip install numpy>=1.21.0", "Installing numpy"):
        print("Trying alternative numpy installation...")
        run_command("pip install numpy", "Installing numpy (latest version)")
    
    # Install other dependencies
    dependencies = [
        ("pyaudio>=0.2.11", "Installing PyAudio"),
        ("librosa>=0.10.0", "Installing librosa"),
        ("soundfile>=0.12.0", "Installing soundfile"),
        ("pandas>=1.3.0", "Installing pandas"),
        ("scikit-learn>=1.0.0", "Installing scikit-learn"),
        ("matplotlib>=3.5.0", "Installing matplotlib"),
    ]
    
    for dep, desc in dependencies:
        if not run_command(f"pip install {dep}", desc):
            print(f"Warning: {desc} failed, trying without version constraint...")
            dep_name = dep.split(">=")[0]
            run_command(f"pip install {dep_name}", f"Installing {dep_name}")
    
    # Install TensorFlow separately (can be problematic)
    print("\nInstalling TensorFlow...")
    if not run_command("pip install tensorflow>=2.10.0", "Installing TensorFlow"):
        print("Trying alternative TensorFlow installation...")
        run_command("pip install tensorflow", "Installing TensorFlow (latest version)")

def install_with_conda():
    """Install dependencies using conda."""
    print("\n" + "="*50)
    print("INSTALLING DEPENDENCIES WITH CONDA")
    print("="*50)
    
    # Update conda
    run_command("conda update conda -y", "Updating conda")
    
    # Install packages with conda
    conda_packages = [
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "librosa",
        "soundfile"
    ]
    
    for package in conda_packages:
        run_command(f"conda install {package} -y", f"Installing {package}")
    
    # Install PyAudio with conda-forge
    run_command("conda install -c conda-forge pyaudio -y", "Installing PyAudio")
    
    # Install TensorFlow
    run_command("conda install tensorflow -y", "Installing TensorFlow")

def install_pyaudio_windows():
    """Install PyAudio specifically for Windows."""
    print("\n" + "="*50)
    print("INSTALLING PYAUDIO FOR WINDOWS")
    print("="*50)
    
    # Try different methods for PyAudio on Windows
    methods = [
        ("pip install pipwin", "Installing pipwin"),
        ("pipwin install pyaudio", "Installing PyAudio with pipwin"),
    ]
    
    for command, desc in methods:
        if run_command(command, desc):
            return True
    
    print("\nAlternative PyAudio installation methods:")
    print("1. Download PyAudio wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
    print("2. Install with: pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl")
    print("3. Or use: conda install -c conda-forge pyaudio")
    
    return False

def verify_installation():
    """Verify that all dependencies are installed correctly."""
    print("\n" + "="*50)
    print("VERIFYING INSTALLATION")
    print("="*50)
    
    packages = [
        "pyaudio",
        "librosa", 
        "soundfile",
        "numpy",
        "pandas",
        "sklearn",
        "tensorflow",
        "matplotlib"
    ]
    
    all_installed = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            all_installed = False
    
    return all_installed

def main():
    """Main installation function."""
    print("SOUND DETECTION APP - DEPENDENCY INSTALLER")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if we're on Windows
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        print(f"✓ Detected Windows system: {platform.platform()}")
    else:
        print(f"✓ Detected system: {platform.platform()}")
    
    # Try pip installation first
    print("\nAttempting pip installation...")
    install_with_pip()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 All dependencies installed successfully!")
        print("\nYou can now run the application:")
        print("python test_recording.py")
    else:
        print("\n⚠ Some dependencies failed to install.")
        
        if is_windows:
            print("\nTrying Windows-specific solutions...")
            install_pyaudio_windows()
            
            print("\nTrying conda installation...")
            install_with_conda()
            
            # Final verification
            if verify_installation():
                print("\n🎉 Dependencies installed successfully!")
            else:
                print("\n❌ Installation incomplete. Please try manual installation:")
                print("1. Install numpy: pip install numpy")
                print("2. Install PyAudio: pip install pipwin && pipwin install pyaudio")
                print("3. Install other packages: pip install librosa soundfile pandas scikit-learn matplotlib")
        else:
            print("\nPlease try manual installation:")
            print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 