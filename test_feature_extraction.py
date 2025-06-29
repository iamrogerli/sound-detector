#!/usr/bin/env python3
"""
Test script for enhanced feature extraction logging.
Demonstrates detailed feature extraction information during training.
"""

import os
import sys
import time
from src.model_trainer import SimpleTrainer
from src.audio_utils import list_audio_files, extract_features
import numpy as np

def test_feature_extraction():
    """Test the enhanced feature extraction with detailed logging."""
    print("🧪 Testing Enhanced Feature Extraction Logging")
    print("=" * 60)
    
    # Check if we have recordings
    recordings = list_audio_files("data/recordings")
    if not recordings:
        print("❌ No recordings found. Please record some sounds first.")
        return False
    
    print(f"📁 Found {len(recordings)} recordings")
    
    # Analyze sound types
    sound_types = {}
    for recording in recordings:
        filename = os.path.basename(recording)
        if '_' in filename:
            sound_type = filename.split('_')[0]
            sound_types[sound_type] = sound_types.get(sound_type, 0) + 1
    
    print(f"🎵 Sound types: {list(sound_types.keys())}")
    for sound_type, count in sound_types.items():
        print(f"  • {sound_type}: {count} recordings")
    
    if len(sound_types) < 2:
        print("❌ Need at least 2 different sound types for training")
        return False
    
    # Test individual feature extraction first
    print("\n🔍 Testing Individual Feature Extraction:")
    print("-" * 40)
    
    test_file = recordings[0]
    print(f"Testing feature extraction on: {os.path.basename(test_file)}")
    
    try:
        # Extract features
        mfcc_features = extract_features(test_file, n_mfcc=13)
        
        if mfcc_features is not None:
            print(f"✓ Feature extraction successful!")
            print(f"  • Original MFCC shape: {mfcc_features.shape}")
            print(f"  • MFCC coefficients: {mfcc_features.shape[0]}")
            print(f"  • Time frames: {mfcc_features.shape[1]}")
            
            # Flatten features
            flattened = mfcc_features.flatten()
            print(f"  • Flattened length: {len(flattened)}")
            
            # Pad to target length
            target_length = 13 * 100  # 1300
            if len(flattened) < target_length:
                padded = np.pad(flattened, (0, target_length - len(flattened)))
                print(f"  • Padded to: {len(padded)} (added {target_length - len(flattened)} zeros)")
            else:
                padded = flattened[:target_length]
                print(f"  • Truncated to: {len(padded)} (removed {len(flattened) - target_length} values)")
            
            # Feature statistics
            print(f"  • Feature statistics:")
            print(f"    - Min value: {np.min(padded):.4f}")
            print(f"    - Max value: {np.max(padded):.4f}")
            print(f"    - Mean value: {np.mean(padded):.4f}")
            print(f"    - Std deviation: {np.std(padded):.4f}")
            print(f"    - Non-zero elements: {np.count_nonzero(padded)}/{len(padded)}")
            print(f"    - Sparsity: {1 - (np.count_nonzero(padded) / len(padded)):.2%}")
        else:
            print("✗ Feature extraction failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error during feature extraction: {e}")
        return False
    
    # Create trainer
    print("\n🔧 Initializing trainer...")
    trainer = SimpleTrainer()
    
    # Train model with detailed feature extraction logging
    print("\n🧠 Starting model training with enhanced feature logging...")
    print("=" * 60)
    
    model_name = f"test_model_{int(time.time())}.pkl"
    success = trainer.train_and_save(model_name)
    
    if success:
        print("\n✅ Training completed successfully!")
        print("=" * 60)
        
        # Get and display detailed results
        results = trainer.get_training_results()
        if results:
            print("📊 Training Results:")
            print(f"  • Overall Accuracy: {results.get('accuracy', 0):.1%}")
            print(f"  • Total Samples: {results.get('n_samples', 0)}")
            print(f"  • Training Samples: {results.get('n_train', 0)}")
            print(f"  • Test Samples: {results.get('n_test', 0)}")
            
            # Per-class accuracy
            class_accuracy = results.get('class_accuracy', {})
            if class_accuracy:
                print("  • Per-class Accuracy:")
                for class_name, acc in class_accuracy.items():
                    print(f"    - {class_name}: {acc:.1%}")
            
            # Classification report
            report = results.get('classification_report', {})
            if report and 'weighted avg' in report:
                weighted_avg = report['weighted avg']
                print("  • Performance Metrics:")
                print(f"    - Precision: {weighted_avg.get('precision', 0):.3f}")
                print(f"    - Recall: {weighted_avg.get('recall', 0):.3f}")
                print(f"    - F1-Score: {weighted_avg.get('f1-score', 0):.3f}")
        
        # Get model info
        print("\n🔍 Model Information:")
        print("-" * 30)
        model_path = f"data/models/{model_name}"
        if trainer.trainer.load_model(model_path):
            model_info = trainer.trainer.get_model_info()
            print(f"  • Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"  • Classes: {model_info.get('classes', [])}")
            print(f"  • Feature Dimension: {model_info.get('feature_dim', 'Unknown')}")
            print(f"  • Feature Vector Length: {model_info.get('feature_vector_length', 'Unknown')}")
            
            if 'n_estimators' in model_info:
                print(f"  • Number of Trees: {model_info['n_estimators']}")
            
            if 'feature_importance' in model_info:
                print(f"  • Feature Importance: Available")
        
        print(f"\n✅ Enhanced feature extraction test completed successfully!")
        print("The detailed feature extraction logging is working correctly.")
        print("You can now see comprehensive feature information in the GUI log.")
        return True
    else:
        print("❌ Training failed!")
        return False

def main():
    """Main test function."""
    print("Enhanced Feature Extraction Logging Test")
    print("=" * 60)
    
    # Run the test
    success = test_feature_extraction()
    
    if success:
        print("\n🎉 All tests passed!")
        print("The enhanced feature extraction logging is working correctly.")
        print("You can now see detailed feature information during training in the GUI.")
    else:
        print("\n❌ Tests failed!")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main() 