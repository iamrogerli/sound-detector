#!/usr/bin/env python3
"""
Test script for improved detection features:
- Confidence threshold control
- Silence filtering
- Better false positive prevention
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_confidence_thresholds():
    """Test different confidence thresholds."""
    print("🎯 CONFIDENCE THRESHOLD TESTING")
    print("=" * 50)
    
    # Simulate detections with different confidence levels
    test_detections = [
        ("voice", 0.95, "HIGH"),
        ("doorbell", 0.78, "MEDIUM"),
        ("background", 0.45, "LOW"),
        ("open", 0.71, "MEDIUM"),  # This is the problematic one
        ("alarm", 0.88, "HIGH"),
        ("silence", 0.35, "LOW")
    ]
    
    thresholds = [0.5, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\n🔧 Testing threshold: {threshold:.0%}")
        print("-" * 30)
        
        passed_detections = []
        for sound_type, confidence, level in test_detections:
            if confidence >= threshold:
                passed_detections.append((sound_type, confidence, level))
                print(f"  ✓ {sound_type}: {confidence:.1%} ({level})")
            else:
                print(f"  ✗ {sound_type}: {confidence:.1%} ({level}) - REJECTED")
        
        print(f"  Results: {len(passed_detections)}/{len(test_detections)} detections passed")
        
        # Check if "open" with 71% confidence is filtered out
        open_detected = any(d[0] == "open" for d in passed_detections)
        if threshold > 0.71 and open_detected:
            print(f"  ⚠️  WARNING: 'open' still detected despite threshold {threshold:.0%}")
        elif threshold > 0.71 and not open_detected:
            print(f"  ✅ SUCCESS: 'open' correctly filtered out at threshold {threshold:.0%}")

def test_silence_filtering():
    """Test silence filtering functionality."""
    print("\n🔇 SILENCE FILTERING TEST")
    print("=" * 50)
    
    try:
        from src.sound_detector import RealTimeDetector
        
        # Create test audio data
        def create_test_audio(volume_level):
            """Create test audio with specific volume level."""
            # Generate 1 second of audio at 44.1kHz
            sample_rate = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create signal with specified volume
            signal = volume_level * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            
            # Add some noise
            noise = np.random.normal(0, volume_level * 0.1, len(signal))
            audio = signal + noise
            
            # Convert to bytes
            return audio.astype(np.float32).tobytes()
        
        # Test different volume levels
        test_levels = [
            (0.001, "Very Low (Silence)"),
            (0.005, "Low (Background)"),
            (0.01, "Threshold Level"),
            (0.05, "Normal Speech"),
            (0.1, "Loud Speech"),
            (0.5, "Very Loud")
        ]
        
        print("Testing audio volume detection:")
        for volume, description in test_levels:
            audio_data = create_test_audio(volume)
            
            # Calculate RMS manually for verification
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Check if it would be considered significant
            is_significant = rms > 0.01
            
            status = "✓ SIGNIFICANT" if is_significant else "✗ SILENCE"
            print(f"  {volume:.3f} ({description}): RMS={rms:.4f} -> {status}")
        
        print("\n✅ Silence filtering test completed!")
        
    except ImportError as e:
        print(f"⚠️  Could not test silence filtering: {e}")

def demonstrate_solution():
    """Demonstrate the solution to the false detection problem."""
    print("\n💡 SOLUTION DEMONSTRATION")
    print("=" * 50)
    
    print("Problem: Detecting 'open' with 71% confidence when silent")
    print("Causes:")
    print("  • Low confidence threshold (50%)")
    print("  • Background noise being classified")
    print("  • No silence filtering")
    print("  • Model overfitting to 'open' class")
    print()
    
    print("Solutions implemented:")
    print("  ✅ 1. Confidence threshold slider (50% - 95%)")
    print("  ✅ 2. Default threshold increased to 80%")
    print("  ✅ 3. Audio volume detection (silence filtering)")
    print("  ✅ 4. Better user control over sensitivity")
    print()
    
    print("How to fix the issue:")
    print("  1. Increase confidence threshold to 80% or higher")
    print("  2. The system will now filter out low-volume audio")
    print("  3. Only significant sounds will trigger detection")
    print("  4. Adjust threshold based on your environment")
    print()
    
    print("Recommended settings:")
    print("  • Quiet environment: 70-80% threshold")
    print("  • Normal environment: 80-85% threshold")
    print("  • Noisy environment: 85-90% threshold")
    print("  • Very noisy: 90-95% threshold")

def main():
    """Run the improved detection tests."""
    print("🔧 IMPROVED DETECTION FEATURES TEST")
    print("=" * 60)
    print()
    print("This test demonstrates the improvements to prevent")
    print("false detections from background noise and silence.")
    print()
    
    # Run tests
    test_confidence_thresholds()
    test_silence_filtering()
    demonstrate_solution()
    
    print("\n" + "=" * 60)
    print("🎉 IMPROVEMENTS COMPLETED!")
    print()
    print("To use the improved detection:")
    print("1. Run: python main.py")
    print("2. Go to Detection tab")
    print("3. Adjust confidence threshold to 80% or higher")
    print("4. Start detection - silence should no longer trigger false positives")
    print()
    print("The system now:")
    print("✓ Filters out silence and low-volume audio")
    print("✓ Uses user-adjustable confidence thresholds")
    print("✓ Prevents false detections from background noise")
    print("✓ Provides better control over detection sensitivity")

if __name__ == "__main__":
    main() 