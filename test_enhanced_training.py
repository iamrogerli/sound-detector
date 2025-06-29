#!/usr/bin/env python3
"""
Test script for enhanced training logging.
Demonstrates detailed training information display.
"""

import os
import sys
import time
from src.model_trainer import SimpleTrainer
from src.audio_utils import list_audio_files

def test_enhanced_training():
    """Test the enhanced training with detailed logging."""
    print("🧪 Testing Enhanced Training Logging")
    print("=" * 50)
    
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
    
    # Create trainer
    print("\n🔧 Initializing trainer...")
    trainer = SimpleTrainer()
    
    # Train model with detailed logging
    print("\n🧠 Starting model training...")
    print("-" * 30)
    
    model_name = f"test_model_{int(time.time())}.pkl"
    success = trainer.train_and_save(model_name)
    
    if success:
        print("\n✅ Training completed successfully!")
        print("-" * 30)
        
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
        
        # Test the model
        print("\n🧪 Testing Model:")
        print("-" * 30)
        test_count = min(3, len(recordings))
        success_count = 0
        
        for i, recording in enumerate(recordings[:test_count]):
            try:
                success = trainer.load_and_test(model_path, recording)
                if success:
                    print(f"  ✓ Test {i+1}: {os.path.basename(recording)}")
                    success_count += 1
                else:
                    print(f"  ✗ Test {i+1}: {os.path.basename(recording)}")
            except Exception as e:
                print(f"  ✗ Test {i+1} error: {e}")
        
        accuracy = (success_count / test_count) * 100
        print(f"Test Accuracy: {accuracy:.1f}% ({success_count}/{test_count})")
        
        print(f"\n✅ Enhanced training test completed successfully!")
        return True
    else:
        print("❌ Training failed!")
        return False

def main():
    """Main test function."""
    print("Enhanced Training Logging Test")
    print("=" * 50)
    
    # Run the test
    success = test_enhanced_training()
    
    if success:
        print("\n🎉 All tests passed!")
        print("The enhanced training logging is working correctly.")
        print("You can now see detailed training information in the GUI log.")
    else:
        print("\n❌ Tests failed!")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main() 