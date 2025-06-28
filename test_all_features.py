#!/usr/bin/env python3
"""
Comprehensive test script for all features of the Sound Detection App.
Tests recording, training, and detection functionality.
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_feature_1_recording():
    """Test Feature 1: Audio Recording"""
    print("\n" + "="*60)
    print("FEATURE 1: AUDIO RECORDING")
    print("="*60)
    
    try:
        from src.audio_recorder import SimpleRecorder, RecordingManager
        from src.audio_utils import AudioPlayer
        
        print("✓ Successfully imported recording modules")
        
        # Test simple recorder
        print("\n--- Testing Simple Recorder ---")
        recorder = SimpleRecorder()
        
        # Test 5-second recording
        test_filename_5 = "data/recordings/feature1_test_5sec.wav"
        print(f"Recording 5 seconds to: {test_filename_5}")
        
        success_5 = recorder.record_5_seconds(test_filename_5)
        
        if success_5:
            print("✓ 5-second recording completed successfully!")
        else:
            print("✗ 5-second recording failed!")
        
        # Test 3-second recording
        test_filename_3 = "data/recordings/feature1_test_3sec.wav"
        print(f"\nRecording 3 seconds to: {test_filename_3}")
        
        success_3 = recorder.record_3_seconds(test_filename_3)
        
        if success_3:
            print("✓ 3-second recording completed successfully!")
        else:
            print("✗ 3-second recording failed!")
        
        # Test playback
        if success_5 or success_3:
            print("\n--- Testing Playback ---")
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
        
        # Test recording manager
        print("\n--- Testing Recording Manager ---")
        manager = RecordingManager()
        
        # List recordings
        recordings = manager.list_recordings()
        print(f"Found {len(recordings)} recordings")
        
        manager.cleanup()
        
        return success_5 or success_3
        
    except Exception as e:
        print(f"✗ Feature 1 test failed: {e}")
        return False


def test_feature_2_training():
    """Test Feature 2: Model Training"""
    print("\n" + "="*60)
    print("FEATURE 2: MODEL TRAINING")
    print("="*60)
    
    try:
        from src.model_trainer import SimpleTrainer
        from src.audio_utils import list_audio_files
        
        print("✓ Successfully imported training modules")
        
        # Check if we have recordings to train on
        recordings = list_audio_files("data/recordings")
        
        if len(recordings) < 2:
            print("⚠ Need at least 2 recordings to train a model")
            print("Please record some sounds first using Feature 1")
            return False
            
        print(f"Found {len(recordings)} recordings for training")
        
        # Test training
        print("\n--- Testing Model Training ---")
        trainer = SimpleTrainer()
        
        # Train a model
        model_name = "feature2_test_model.pkl"
        print(f"Training model: {model_name}")
        
        success = trainer.train_and_save(model_name)
        
        if success:
            print("✓ Feature 2: Model training completed successfully!")
            
            # Test the model
            print("\n--- Testing Model Prediction ---")
            test_file = recordings[0]  # Use first recording as test
            test_success = trainer.load_and_test(
                f"data/models/{model_name}", 
                test_file
            )
            
            if test_success:
                print("✓ Model prediction test successful!")
            else:
                print("✗ Model prediction test failed!")
                
        else:
            print("✗ Feature 2: Model training failed!")
            
        return success
        
    except Exception as e:
        print(f"✗ Feature 2 test failed: {e}")
        return False


def test_feature_3_detection():
    """Test Feature 3: Real-time Detection"""
    print("\n" + "="*60)
    print("FEATURE 3: REAL-TIME DETECTION")
    print("="*60)
    
    try:
        from src.sound_detector import SimpleDetector
        from src.audio_utils import list_audio_files
        
        print("✓ Successfully imported detection modules")
        
        # Check if we have a trained model
        model_path = "data/models/feature2_test_model.pkl"
        
        if not os.path.exists(model_path):
            print("⚠ No trained model found")
            print("Please train a model first using Feature 2")
            return False
            
        print(f"Found trained model: {model_path}")
        
        # Test detector
        print("\n--- Testing Sound Detector ---")
        detector = SimpleDetector(model_path)
        
        # Test detection on a file
        recordings = list_audio_files("data/recordings")
        if recordings:
            test_file = recordings[0]
            print(f"Testing detection on: {test_file}")
            
            test_success = detector.test_detection(test_file)
            
            if test_success:
                print("✓ Feature 3: Detection test successful!")
                
                # Test real-time monitoring (short duration)
                print("\n--- Testing Real-time Monitoring (5 seconds) ---")
                print("This will monitor for 5 seconds. Make some noise!")
                
                try:
                    detector.start_monitoring("test", duration=5)
                    print("✓ Real-time monitoring test completed!")
                except KeyboardInterrupt:
                    print("\nMonitoring stopped by user")
                    
            else:
                print("✗ Feature 3: Detection test failed!")
                
        else:
            print("⚠ No recordings found for testing")
            
        detector.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Feature 3 test failed: {e}")
        return False


def main():
    """Main test function."""
    print("SOUND DETECTION APP - COMPREHENSIVE FEATURE TEST")
    print("="*60)
    
    # Ensure directories exist
    os.makedirs("data/recordings", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Test results
    results = {
        'feature1': False,
        'feature2': False,
        'feature3': False
    }
    
    # Test each feature
    results['feature1'] = test_feature_1_recording()
    
    if results['feature1']:
        results['feature2'] = test_feature_2_training()
        
        if results['feature2']:
            results['feature3'] = test_feature_3_detection()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"Feature 1 (Recording):     {'✓ PASS' if results['feature1'] else '✗ FAIL'}")
    print(f"Feature 2 (Training):      {'✓ PASS' if results['feature2'] else '✗ FAIL'}")
    print(f"Feature 3 (Detection):     {'✓ PASS' if results['feature3'] else '✗ FAIL'}")
    
    if all(results.values()):
        print("\n🎉 ALL FEATURES WORKING! 🎉")
        print("Your Sound Detection App is ready to use!")
        print("\nTo run the full GUI application:")
        print("python main.py")
    else:
        print("\n⚠ Some features need attention.")
        print("Check the error messages above and ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main() 