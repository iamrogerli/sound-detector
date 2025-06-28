#!/usr/bin/env python3
"""
Test script for press-and-hold recording functionality.
"""

import os
import time
from src.audio_recorder import SimpleRecorder

def test_hold_recording():
    """Test the press-and-hold recording functionality."""
    print("Testing press-and-hold recording functionality...")
    
    # Ensure recordings directory exists
    os.makedirs("data/recordings", exist_ok=True)
    
    # Create recorder
    recorder = SimpleRecorder()
    
    try:
        # Test 1: Start recording
        print("\n1. Starting manual recording...")
        recorder.start_recording()
        
        # Check if recording started
        if recorder.is_recording():
            print("✓ Recording started successfully")
        else:
            print("✗ Failed to start recording")
            return False
        
        # Test 2: Record for a few seconds
        print("2. Recording for 3 seconds...")
        time.sleep(3)
        
        # Test 3: Stop recording
        print("3. Stopping recording...")
        recorder.stop_recording()
        
        # Wait a moment for the recording thread to finish
        time.sleep(0.5)
        
        if not recorder.is_recording():
            print("✓ Recording stopped successfully")
        else:
            print("✗ Failed to stop recording")
            return False
        
        # Test 4: Save recording
        print("4. Saving recording...")
        filename = "data/recordings/test_hold_recording.wav"
        success = recorder.save_recording(filename)
        
        if success:
            print(f"✓ Recording saved to: {filename}")
            
            # Check if file exists and has content
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                print("✓ File exists and has content")
                return True
            else:
                print("✗ File doesn't exist or is empty")
                return False
        else:
            print("✗ Failed to save recording")
            return False
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
    finally:
        recorder.cleanup()

def test_multiple_recordings():
    """Test multiple press-and-hold recordings."""
    print("\nTesting multiple press-and-hold recordings...")
    
    recorder = SimpleRecorder()
    
    try:
        for i in range(3):
            print(f"\nRecording {i+1}:")
            
            # Start recording
            recorder.start_recording()
            time.sleep(1)  # Record for 1 second
            
            # Stop recording
            recorder.stop_recording()
            time.sleep(0.5)
            
            # Save recording
            filename = f"data/recordings/test_hold_{i+1}.wav"
            success = recorder.save_recording(filename)
            
            if success:
                print(f"✓ Recording {i+1} saved")
            else:
                print(f"✗ Failed to save recording {i+1}")
                return False
        
        print("✓ All multiple recordings completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error during multiple recordings test: {e}")
        return False
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    print("Press-and-Hold Recording Test")
    print("=" * 40)
    
    # Test single recording
    success1 = test_hold_recording()
    
    # Test multiple recordings
    success2 = test_multiple_recordings()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("✓ All tests passed! Press-and-hold recording works correctly.")
    else:
        print("✗ Some tests failed. Check the output above for details.") 