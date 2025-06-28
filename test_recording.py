#!/usr/bin/env python3
"""
Simple test script for the audio recording functionality.
Tests the basic recording features without the full GUI.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.audio_recorder import SimpleRecorder
    from src.audio_utils import AudioPlayer
    print("✓ Successfully imported audio modules")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)


def test_recording():
    """Test the basic recording functionality."""
    print("\n=== Testing Audio Recording ===")
    
    # Create recorder
    recorder = SimpleRecorder()
    
    # Test recording
    test_filename = "data/recordings/test_recording.wav"
    print(f"Recording 5 seconds to: {test_filename}")
    
    try:
        success = recorder.record_5_seconds(test_filename)
        
        if success:
            print("✓ Recording completed successfully!")
            
            # Test playback
            print("\n=== Testing Audio Playback ===")
            player = AudioPlayer()
            print("Playing back the recording...")
            player.play_file(test_filename)
            player.cleanup()
            print("✓ Playback completed!")
            
        else:
            print("✗ Recording failed!")
            
    except Exception as e:
        print(f"✗ Error during recording: {e}")
        
    finally:
        recorder.cleanup()


def test_recording_manager():
    """Test the recording manager functionality."""
    print("\n=== Testing Recording Manager ===")
    
    try:
        from src.audio_recorder import RecordingManager
        
        manager = RecordingManager()
        
        # List recordings
        recordings = manager.list_recordings()
        print(f"Found {len(recordings)} existing recordings")
        
        # Get info about recordings
        for recording in recordings[:3]:  # Show first 3
            info = manager.get_recording_info(recording)
            print(f"  - {os.path.basename(recording)}: {info['duration']:.1f}s, {info['size']/1024:.1f}KB")
            
        manager.cleanup()
        print("✓ Recording manager test completed!")
        
    except Exception as e:
        print(f"✗ Error testing recording manager: {e}")


def main():
    """Main test function."""
    print("Sound Detection App - Basic Functionality Test")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists("data/recordings"):
        os.makedirs("data/recordings", exist_ok=True)
        print("✓ Created data/recordings directory")
    
    # Run tests
    test_recording()
    test_recording_manager()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo run the full application with GUI:")
    print("python main.py")


if __name__ == "__main__":
    main() 