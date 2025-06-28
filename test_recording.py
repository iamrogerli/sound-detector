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
    
    # Test 5-second recording
    test_filename_5 = "data/recordings/test_sound_5sec.wav"
    print(f"Recording 5 seconds to: {test_filename_5}")
    
    try:
        success_5 = recorder.record_5_seconds(test_filename_5)
        
        if success_5:
            print("✓ 5-second recording completed successfully!")
        else:
            print("✗ 5-second recording failed!")
            
    except Exception as e:
        print(f"✗ Error during 5-second recording: {e}")
        success_5 = False
    
    # Test 3-second recording
    test_filename_3 = "data/recordings/test_sound_3sec.wav"
    print(f"\nRecording 3 seconds to: {test_filename_3}")
    
    try:
        success_3 = recorder.record_3_seconds(test_filename_3)
        
        if success_3:
            print("✓ 3-second recording completed successfully!")
        else:
            print("✗ 3-second recording failed!")
            
    except Exception as e:
        print(f"✗ Error during 3-second recording: {e}")
        success_3 = False
    
    # Test playback if either recording succeeded
    if success_5 or success_3:
        print("\n=== Testing Audio Playback ===")
        player = AudioPlayer()
        
        if success_5:
            print("Playing back the 5-second recording...")
            player.play_file(test_filename_5)
            
        if success_3:
            print("Playing back the 3-second recording...")
            player.play_file(test_filename_3)
            
        player.cleanup()
        print("✓ Playback completed!")
        
    recorder.cleanup()
    return success_5 or success_3


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