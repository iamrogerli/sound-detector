"""
Sound detector module for the sound detection app.
Handles real-time sound detection from microphone input.
"""

import pyaudio
import numpy as np
import wave
import threading
import time
import os
from typing import Optional, Callable, Dict, List
from .audio_utils import AudioConfig, extract_features
from .model_trainer import SoundModelTrainer


class RealTimeDetector:
    """Real-time sound detection from microphone."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.audio = pyaudio.PyAudio()
        self.is_detecting = False
        self.detection_thread = None
        self.model_trainer = SoundModelTrainer()
        self.on_detection: Optional[Callable] = None
        self.detection_history: List[Dict] = []
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model_trainer.load_model(model_path)
            
    def start_detection(self, target_sounds: Optional[List[str]] = None,
                       confidence_threshold: float = 0.7,
                       detection_interval: float = 2.0) -> bool:
        """
        Start real-time sound detection.
        
        Args:
            target_sounds: List of sound types to detect (None for all)
            confidence_threshold: Minimum confidence for detection
            detection_interval: Time between detections in seconds
            
        Returns:
            True if detection started successfully
        """
        if self.is_detecting:
            print("Detection already running")
            return False
            
        if self.model_trainer.model is None:
            print("No model loaded for detection")
            return False
            
        self.is_detecting = True
        self.target_sounds = target_sounds
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        
        # Start detection in separate thread
        self.detection_thread = threading.Thread(
            target=self._detection_loop
        )
        self.detection_thread.start()
        
        print("Real-time detection started")
        return True
        
    def stop_detection(self) -> None:
        """Stop real-time sound detection."""
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join()
        print("Real-time detection stopped")
        
    def _detection_loop(self) -> None:
        """Main detection loop."""
        while self.is_detecting:
            try:
                # Record a short audio segment
                audio_data = self._record_audio_segment()
                
                if audio_data is not None:
                    # Save temporary file for analysis
                    temp_file = "temp_detection.wav"
                    self._save_audio_segment(audio_data, temp_file)
                    
                    # Analyze the audio
                    prediction, confidence = self.model_trainer.predict_sound(temp_file)
                    
                    # Check if detection meets criteria
                    if (confidence >= self.confidence_threshold and 
                        (self.target_sounds is None or prediction in self.target_sounds)):
                        
                        detection_info = {
                            'timestamp': time.time(),
                            'sound_type': prediction,
                            'confidence': confidence,
                            'audio_file': temp_file
                        }
                        
                        self.detection_history.append(detection_info)
                        
                        print(f"Detection: {prediction} (confidence: {confidence:.3f})")
                        
                        # Call callback if provided
                        if self.on_detection:
                            self.on_detection(detection_info)
                    
                    # Clean up temp file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
            except Exception as e:
                print(f"Error in detection loop: {e}")
                
            # Wait before next detection
            time.sleep(self.detection_interval)
            
    def _record_audio_segment(self, duration: float = 3.0) -> Optional[bytes]:
        """
        Record a short audio segment.
        
        Args:
            duration: Duration to record in seconds
            
        Returns:
            Audio data as bytes
        """
        try:
            stream = self.audio.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            
            frames = []
            num_chunks = int(self.config.RATE / self.config.CHUNK * duration)
            
            for _ in range(num_chunks):
                if not self.is_detecting:
                    break
                data = stream.read(self.config.CHUNK)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            return b''.join(frames)
            
        except Exception as e:
            print(f"Error recording audio segment: {e}")
            return None
            
    def _save_audio_segment(self, audio_data: bytes, filename: str) -> None:
        """Save audio segment to temporary file."""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.config.FORMAT))
                wf.setframerate(self.config.RATE)
                wf.writeframes(audio_data)
        except Exception as e:
            print(f"Error saving audio segment: {e}")
            
    def get_detection_history(self, limit: int = 10) -> List[Dict]:
        """Get recent detection history."""
        return self.detection_history[-limit:]
        
    def clear_detection_history(self) -> None:
        """Clear detection history."""
        self.detection_history.clear()
        
    def load_model(self, model_path: str) -> bool:
        """Load a trained model for detection."""
        return self.model_trainer.load_model(model_path)
        
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return self.model_trainer.get_model_info()
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_detection()
        self.audio.terminate()


class SimpleDetector:
    """Simplified detector for easy use."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.detector = RealTimeDetector(model_path)
        
    def start_monitoring(self, target_sound: str, duration: int = 30) -> None:
        """
        Start monitoring for a specific sound.
        
        Args:
            target_sound: Sound type to monitor for
            duration: How long to monitor in seconds
        """
        print(f"Starting to monitor for '{target_sound}' for {duration} seconds...")
        
        # Set up detection callback
        def on_detection(detection_info):
            print(f"🎵 DETECTED: {detection_info['sound_type']} "
                  f"(confidence: {detection_info['confidence']:.3f})")
        
        self.detector.on_detection = on_detection
        
        # Start detection
        if self.detector.start_detection(target_sounds=[target_sound]):
            try:
                time.sleep(duration)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
            finally:
                self.detector.stop_detection()
        else:
            print("Failed to start detection")
            
    def test_detection(self, test_file: str) -> bool:
        """
        Test detection on a recorded audio file.
        
        Args:
            test_file: Path to test audio file
            
        Returns:
            True if detection was successful
        """
        try:
            prediction, confidence = self.detector.model_trainer.predict_sound(test_file)
            print(f"Test result: {prediction} (confidence: {confidence:.3f})")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self.detector.cleanup()


class DetectionLogger:
    """Logs detection events to file."""
    
    def __init__(self, log_file: str = "detection_log.txt"):
        self.log_file = log_file
        
    def log_detection(self, detection_info: Dict) -> None:
        """Log a detection event."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", 
                                time.localtime(detection_info['timestamp']))
        
        log_entry = (f"{timestamp} - {detection_info['sound_type']} "
                    f"(confidence: {detection_info['confidence']:.3f})\n")
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging detection: {e}")
            
    def get_recent_logs(self, lines: int = 20) -> List[str]:
        """Get recent log entries."""
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading logs: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = SimpleDetector()
    
    # Test with a model if available
    model_path = "data/models/sound_model.pkl"
    if os.path.exists(model_path):
        detector.detector.load_model(model_path)
        print("Model loaded successfully")
        
        # Test detection on a file
        test_files = [f for f in os.listdir("data/recordings") 
                     if f.endswith('.wav')]
        if test_files:
            test_file = os.path.join("data/recordings", test_files[0])
            detector.test_detection(test_file)
    else:
        print("No model found. Please train a model first.")
        
    detector.cleanup() 