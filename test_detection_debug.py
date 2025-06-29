#!/usr/bin/env python3
"""
Debug script for real-time detection issues.
Tests feature extraction consistency and model predictions.
"""

import os
import sys
import time
import numpy as np
from src.model_trainer import SimpleTrainer
from src.audio_utils import list_audio_files, extract_features
from src.sound_detector import RealTimeDetector

def test_feature_consistency():
    """Test if feature extraction is consistent between training and detection."""
    print("🔍 Testing Feature Extraction Consistency")
    print("=" * 50)
    
    # Get recordings
    recordings = list_audio_files("data/recordings")
    open_recordings = [r for r in recordings if 'open' in os.path.basename(r)]
    
    if not open_recordings:
        print("❌ No 'open' recordings found")
        return False
    
    print(f"Found {len(open_recordings)} 'open' recordings")
    
    # Test feature extraction on first few open recordings
    for i, recording in enumerate(open_recordings[:3]):
        print(f"\n📁 Testing: {os.path.basename(recording)}")
        
        try:
            # Extract features
            mfcc_features = extract_features(recording, n_mfcc=13)
            
            if mfcc_features is not None:
                print(f"  • Original shape: {mfcc_features.shape}")
                
                # Flatten and pad like in training
                flattened = mfcc_features.flatten()
                target_length = 13 * 100
                
                if len(flattened) < target_length:
                    padded = np.pad(flattened, (0, target_length - len(flattened)))
                    print(f"  • Padded to: {len(padded)}")
                else:
                    padded = flattened[:target_length]
                    print(f"  • Truncated to: {len(padded)}")
                
                # Feature statistics
                print(f"  • Min: {np.min(padded):.4f}, Max: {np.max(padded):.4f}")
                print(f"  • Mean: {np.mean(padded):.4f}, Std: {np.std(padded):.4f}")
                
            else:
                print("  ✗ Feature extraction failed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return True

def test_model_predictions():
    """Test model predictions on existing recordings."""
    print("\n🧠 Testing Model Predictions")
    print("=" * 50)
    
    # Get latest model
    model_files = [f for f in os.listdir("data/models") if f.endswith('.pkl')]
    if not model_files:
        print("❌ No trained models found")
        return False
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join("data/models", latest_model)
    print(f"Using model: {latest_model}")
    
    # Load model
    trainer = SimpleTrainer()
    if not trainer.trainer.load_model(model_path):
        print("❌ Failed to load model")
        return False
    
    # Test on open recordings
    recordings = list_audio_files("data/recordings")
    open_recordings = [r for r in recordings if 'open' in os.path.basename(r)]
    
    print(f"\nTesting on {min(5, len(open_recordings))} 'open' recordings:")
    
    success_count = 0
    for i, recording in enumerate(open_recordings[:5]):
        try:
            prediction, confidence = trainer.trainer.predict_sound(recording)
            filename = os.path.basename(recording)
            
            if prediction == 'open':
                print(f"  ✓ {filename}: {prediction} ({confidence:.3f})")
                success_count += 1
            else:
                print(f"  ✗ {filename}: {prediction} ({confidence:.3f}) - Expected 'open'")
                
        except Exception as e:
            print(f"  ✗ {os.path.basename(recording)}: Error - {e}")
    
    accuracy = (success_count / min(5, len(open_recordings))) * 100
    print(f"\nModel accuracy on 'open' recordings: {accuracy:.1f}%")
    
    return accuracy > 50

def test_real_time_simulation():
    """Simulate real-time detection process."""
    print("\n🎤 Testing Real-time Detection Simulation")
    print("=" * 50)
    
    # Get latest model
    model_files = [f for f in os.listdir("data/models") if f.endswith('.pkl')]
    if not model_files:
        print("❌ No trained models found")
        return False
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join("data/models", latest_model)
    
    # Create detector
    detector = RealTimeDetector(model_path)
    
    # Test on a few open recordings
    recordings = list_audio_files("data/recordings")
    open_recordings = [r for r in recordings if 'open' in os.path.basename(r)]
    
    print(f"Testing real-time detection simulation on {min(3, len(open_recordings))} 'open' recordings:")
    
    success_count = 0
    for i, recording in enumerate(open_recordings[:3]):
        try:
            # Simulate the detection process
            prediction, confidence = detector.model_trainer.predict_sound(recording)
            filename = os.path.basename(recording)
            
            print(f"  • {filename}:")
            print(f"    - Prediction: {prediction}")
            print(f"    - Confidence: {confidence:.3f}")
            
            # Test different confidence thresholds
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            for threshold in thresholds:
                would_detect = confidence >= threshold
                status = "✓" if would_detect else "✗"
                print(f"    - Threshold {threshold:.1f}: {status}")
            
            if prediction == 'open':
                success_count += 1
                
        except Exception as e:
            print(f"  ✗ {os.path.basename(recording)}: Error - {e}")
    
    print(f"\nReal-time simulation accuracy: {(success_count/3)*100:.1f}%")
    
    detector.cleanup()
    return success_count > 0

def test_audio_volume_detection():
    """Test audio volume detection logic."""
    print("\n🔊 Testing Audio Volume Detection")
    print("=" * 50)
    
    # Get a few recordings
    recordings = list_audio_files("data/recordings")
    test_recordings = recordings[:3]
    
    print("Testing audio volume detection on recordings:")
    
    for recording in test_recordings:
        try:
            # Load audio data
            import librosa
            audio_data, sr = librosa.load(recording, sr=None)
            
            # Calculate RMS volume
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Test volume threshold
            volume_threshold = 0.01
            is_significant = rms > volume_threshold
            
            print(f"  • {os.path.basename(recording)}:")
            print(f"    - RMS: {rms:.6f}")
            print(f"    - Threshold: {volume_threshold}")
            print(f"    - Significant: {'✓' if is_significant else '✗'}")
            
        except Exception as e:
            print(f"  ✗ {os.path.basename(recording)}: Error - {e}")

def main():
    """Main debug function."""
    print("Real-time Detection Debug Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Feature Consistency", test_feature_consistency),
        ("Model Predictions", test_model_predictions),
        ("Real-time Simulation", test_real_time_simulation),
        ("Audio Volume Detection", test_audio_volume_detection)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("DEBUG SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    if not results.get("Model Predictions", False):
        print("• Model is not correctly predicting 'open' sounds")
        print("• Try retraining with more diverse 'open' recordings")
        print("• Check if recordings are clear and consistent")
    
    if not results.get("Real-time Simulation", False):
        print("• Real-time detection simulation failed")
        print("• Check confidence thresholds - try lowering to 0.5")
        print("• Verify audio preprocessing is consistent")
    
    print("• Try lowering the confidence threshold in the GUI to 50-60%")
    print("• Ensure you're speaking clearly and at consistent volume")
    print("• Check microphone settings and permissions")
    
    return all(results.values())

if __name__ == "__main__":
    main() 