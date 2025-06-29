"""
Model trainer module for the sound detection app.
Handles training machine learning models to recognize specific sounds.
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from .audio_utils import extract_features, list_audio_files


class SoundModelTrainer:
    """Trains and manages sound recognition models."""
    
    def __init__(self, recordings_dir: str = "data/recordings", 
                 models_dir: str = "data/models"):
        self.recordings_dir = recordings_dir
        self.models_dir = models_dir
        self.model = None
        self.classes = []
        self.feature_dim = 13  # MFCC features
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from recorded audio files.
        
        Returns:
            Tuple of (features, labels) for training
        """
        print("Preparing training data...")
        
        features = []
        labels = []
        feature_stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'feature_shapes': [],
            'original_lengths': [],
            'padded_lengths': []
        }
        
        # Get all audio files
        audio_files = list_audio_files(self.recordings_dir)
        
        if not audio_files:
            raise ValueError("No audio files found for training")
            
        print(f"Found {len(audio_files)} audio files")
        feature_stats['total_files'] = len(audio_files)
        
        # Process each audio file
        for i, audio_file in enumerate(audio_files):
            try:
                print(f"Processing file {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                
                # Extract features
                mfcc_features = extract_features(audio_file, n_mfcc=self.feature_dim)
                
                if mfcc_features is not None:
                    # Get sound type from filename
                    filename = os.path.basename(audio_file)
                    sound_type = filename.split('_')[0]
                    
                    # Log original feature shape
                    original_shape = mfcc_features.shape
                    feature_stats['feature_shapes'].append(original_shape)
                    print(f"  Original MFCC shape: {original_shape}")
                    
                    # Flatten MFCC features
                    flattened_features = mfcc_features.flatten()
                    original_length = len(flattened_features)
                    feature_stats['original_lengths'].append(original_length)
                    print(f"  Flattened length: {original_length}")
                    
                    # Pad or truncate to fixed length
                    target_length = self.feature_dim * 100  # Adjust based on your needs
                    if len(flattened_features) < target_length:
                        # Pad with zeros
                        padded_features = np.pad(flattened_features, 
                                               (0, target_length - len(flattened_features)))
                        print(f"  Padded to length: {target_length} (added {target_length - original_length} zeros)")
                    else:
                        # Truncate
                        padded_features = flattened_features[:target_length]
                        print(f"  Truncated to length: {target_length} (removed {original_length - target_length} values)")
                    
                    feature_stats['padded_lengths'].append(len(padded_features))
                    
                    # Log feature statistics
                    print(f"  Feature statistics:")
                    print(f"    - Min value: {np.min(padded_features):.4f}")
                    print(f"    - Max value: {np.max(padded_features):.4f}")
                    print(f"    - Mean value: {np.mean(padded_features):.4f}")
                    print(f"    - Std deviation: {np.std(padded_features):.4f}")
                    print(f"    - Non-zero elements: {np.count_nonzero(padded_features)}/{len(padded_features)}")
                    
                    features.append(padded_features)
                    labels.append(sound_type)
                    feature_stats['processed_files'] += 1
                    
                    print(f"  ✓ Processed: {filename} -> {sound_type}")
                    
                else:
                    print(f"  ✗ Failed to extract features from {filename}")
                    feature_stats['failed_files'] += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(audio_file)}: {e}")
                feature_stats['failed_files'] += 1
                continue
                
        if not features:
            raise ValueError("No valid features extracted from audio files")
            
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Get unique classes
        self.classes = sorted(list(set(y)))
        
        # Log overall feature statistics
        print(f"\nFeature Extraction Summary:")
        print(f"  • Total files: {feature_stats['total_files']}")
        print(f"  • Successfully processed: {feature_stats['processed_files']}")
        print(f"  • Failed to process: {feature_stats['failed_files']}")
        print(f"  • Final feature matrix shape: {X.shape}")
        print(f"  • Feature vector length: {X.shape[1]}")
        print(f"  • Classes: {self.classes}")
        
        if feature_stats['feature_shapes']:
            unique_shapes = set(feature_stats['feature_shapes'])
            print(f"  • Original MFCC shapes found: {list(unique_shapes)}")
        
        if feature_stats['original_lengths']:
            print(f"  • Original lengths - Min: {min(feature_stats['original_lengths'])}, Max: {max(feature_stats['original_lengths'])}")
        
        # Log overall feature statistics
        print(f"  • Overall feature statistics:")
        print(f"    - Min value: {np.min(X):.4f}")
        print(f"    - Max value: {np.max(X):.4f}")
        print(f"    - Mean value: {np.mean(X):.4f}")
        print(f"    - Std deviation: {np.std(X):.4f}")
        print(f"    - Sparsity: {1 - (np.count_nonzero(X) / X.size):.2%}")
        
        print(f"Prepared {len(X)} samples with {len(self.classes)} classes: {self.classes}")
        
        return X, y
        
    def train_model(self, model_type: str = "random_forest") -> Dict:
        """
        Train a sound recognition model.
        
        Args:
            model_type: Type of model to train ("random_forest", "svm", etc.)
            
        Returns:
            Dictionary with training results
        """
        print(f"Training {model_type} model...")
        
        # Prepare data
        X, y = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for i, class_name in enumerate(self.classes):
            class_mask = y_test == class_name
            if np.any(class_mask):
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_accuracy[class_name] = class_acc
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_.tolist()
        
        results = {
            'model_type': model_type,
            'accuracy': accuracy,
            'classes': self.classes,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'classification_report': report,
            'class_accuracy': class_accuracy,
            'feature_importance': feature_importance,
            'feature_dim': self.feature_dim,
            'model_params': self.model.get_params() if self.model else None
        }
        
        print(f"Training completed! Accuracy: {accuracy:.3f}")
        print(f"Per-class accuracy: {class_accuracy}")
        
        return results
        
    def save_model(self, filename: str = "sound_model.pkl") -> str:
        """
        Save the trained model to file.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No trained model to save")
            
        filepath = os.path.join(self.models_dir, filename)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'classes': self.classes,
            'feature_dim': self.feature_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to: {filepath}")
        return filepath
        
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.classes = model_data['classes']
            self.feature_dim = model_data.get('feature_dim', 13)
            
            print(f"Model loaded from: {filepath}")
            print(f"Classes: {self.classes}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def predict_sound(self, audio_file: str) -> Tuple[str, float]:
        """
        Predict the sound type for an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        if self.model is None:
            raise ValueError("No model loaded")
            
        # Extract features
        mfcc_features = extract_features(audio_file, n_mfcc=self.feature_dim)
        
        if mfcc_features is None:
            raise ValueError("Could not extract features from audio file")
            
        # Flatten and pad features
        flattened_features = mfcc_features.flatten()
        target_length = self.feature_dim * 100
        
        if len(flattened_features) < target_length:
            padded_features = np.pad(flattened_features, 
                                   (0, target_length - len(flattened_features)))
        else:
            padded_features = flattened_features[:target_length]
            
        # Reshape for prediction
        features = padded_features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        confidence = np.max(self.model.predict_proba(features))
        
        return prediction, confidence
        
    def get_model_info(self) -> Dict:
        """Get comprehensive information about the current model."""
        if self.model is None:
            return {'status': 'No model loaded'}
            
        info = {
            'status': 'Model loaded',
            'model_type': type(self.model).__name__,
            'classes': self.classes,
            'feature_dim': self.feature_dim,
            'n_classes': len(self.classes),
            'feature_vector_length': self.feature_dim * 100
        }
        
        # Add model parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            info['feature_importance'] = self.model.feature_importances_.tolist()
            info['top_features'] = np.argsort(self.model.feature_importances_)[-10:].tolist()
        
        # Add training info if available
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'n_features_in_'):
            info['n_features_in'] = self.model.n_features_in_
        
        return info


class SimpleTrainer:
    """Simplified trainer for quick model training."""
    
    def __init__(self):
        self.trainer = SoundModelTrainer()
        self.last_training_results = None
        
    def train_and_save(self, model_name: str = "sound_model.pkl") -> bool:
        """
        Train a model and save it.
        
        Args:
            model_name: Name for the saved model file
            
        Returns:
            True if training and saving was successful
        """
        try:
            print("Starting model training...")
            
            # Train model
            self.last_training_results = self.trainer.train_model()
            
            # Save model
            self.trainer.save_model(model_name)
            
            print(f"Training completed successfully!")
            print(f"Accuracy: {self.last_training_results['accuracy']:.3f}")
            print(f"Classes: {self.last_training_results['classes']}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def get_training_results(self) -> Dict:
        """Get the results from the last training session."""
        return self.last_training_results or {}
            
    def load_and_test(self, model_path: str, test_file: str) -> bool:
        """
        Load a model and test it on a file.
        
        Args:
            model_path: Path to the model file
            test_file: Path to test audio file
            
        Returns:
            True if test was successful
        """
        try:
            # Load model
            if not self.trainer.load_model(model_path):
                return False
                
            # Test prediction
            prediction, confidence = self.trainer.predict_sound(test_file)
            
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Testing failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Test the trainer
    trainer = SimpleTrainer()
    
    # Train a model
    success = trainer.train_and_save("test_model.pkl")
    
    if success:
        print("Model training test successful!")
        
        # Test the model
        test_files = list_audio_files("data/recordings")
        if test_files:
            trainer.load_and_test("data/models/test_model.pkl", test_files[0])
    else:
        print("Model training test failed!") 