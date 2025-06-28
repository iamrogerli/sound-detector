"""
Audio recording module for the sound detection app.
Handles 5-second audio recordings and file management.
"""

import os
import time
import threading
from datetime import datetime
from typing import Optional, Callable
from .audio_utils import AudioRecorder, AudioConfig, AudioPlayer


class RecordingManager:
    """Manages audio recording operations with file organization."""
    
    def __init__(self, recordings_dir: str = "data/recordings"):
        self.recordings_dir = recordings_dir
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        self.recording_thread = None
        self.on_recording_complete: Optional[Callable] = None
        
        # Ensure recordings directory exists
        os.makedirs(recordings_dir, exist_ok=True)
        
    def start_recording(self, sound_type: str = "unknown") -> Optional[str]:
        """
        Start a 5-second recording for a specific sound type.
        
        Args:
            sound_type: Category/type of sound being recorded
            
        Returns:
            Path to the saved audio file or None if recording failed
        """
        if self.recording_thread and self.recording_thread.is_alive():
            print("Recording already in progress...")
            return None
            
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sound_type}_{timestamp}.wav"
        filepath = os.path.join(self.recordings_dir, filename)
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_and_save,
            args=(filepath,)
        )
        self.recording_thread.start()
        
        return filepath
        
    def _record_and_save(self, filepath: str) -> None:
        """Internal method to record and save audio."""
        try:
            success = self.recorder.record_and_save(filepath)
            if success and self.on_recording_complete:
                self.on_recording_complete(filepath)
        except Exception as e:
            print(f"Error during recording: {e}")
            
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return (self.recording_thread and 
                self.recording_thread.is_alive())
                
    def stop_recording(self) -> None:
        """Stop current recording."""
        self.recorder.stop_recording()
        
    def play_recording(self, filepath: str) -> None:
        """Play a recorded audio file."""
        if os.path.exists(filepath):
            self.player.play_file(filepath)
        else:
            print(f"File not found: {filepath}")
            
    def list_recordings(self, sound_type: Optional[str] = None) -> list:
        """
        List all recordings, optionally filtered by sound type.
        
        Args:
            sound_type: Optional filter for specific sound type
            
        Returns:
            List of recording file paths
        """
        recordings = []
        
        if not os.path.exists(self.recordings_dir):
            return recordings
            
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith('.wav'):
                if sound_type is None or filename.startswith(sound_type):
                    recordings.append(os.path.join(self.recordings_dir, filename))
                    
        return sorted(recordings)
        
    def delete_recording(self, filepath: str) -> bool:
        """Delete a recording file."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Deleted recording: {filepath}")
                return True
            else:
                print(f"File not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
            
    def get_recording_info(self, filepath: str) -> dict:
        """Get information about a recording."""
        info = {
            'filepath': filepath,
            'exists': False,
            'size': 0,
            'duration': 0,
            'created': None
        }
        
        if os.path.exists(filepath):
            info['exists'] = True
            info['size'] = os.path.getsize(filepath)
            info['created'] = datetime.fromtimestamp(os.path.getctime(filepath))
            
            # Get duration using librosa
            try:
                import librosa
                y, sr = librosa.load(filepath, sr=None)
                info['duration'] = librosa.get_duration(y=y, sr=sr)
            except ImportError:
                print("librosa not available, duration will be 0")
                info['duration'] = 0
            except Exception as e:
                print(f"Error getting duration: {e}")
                info['duration'] = 0
                
        return info
        
    def cleanup(self):
        """Clean up resources."""
        self.recorder.cleanup()
        self.player.cleanup()


class SimpleRecorder:
    """Simplified recorder for quick 5-second recordings."""
    
    def __init__(self):
        self.recorder = AudioRecorder()
        
    def record_5_seconds(self, filename: str) -> bool:
        """
        Record exactly 5 seconds of audio and save to file.
        
        Args:
            filename: Path where to save the audio file
            
        Returns:
            True if recording was successful
        """
        print("Starting 5-second recording...")
        print("Recording will start in 3 seconds...")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print("Recording now!")
        success = self.recorder.record_and_save(filename)
        
        if success:
            print(f"Recording completed and saved to: {filename}")
        else:
            print("Recording failed!")
            
        return bool(success)
        
    def record_3_seconds(self, filename: str) -> bool:
        """
        Record exactly 3 seconds of audio and save to file.
        
        Args:
            filename: Path where to save the audio file
            
        Returns:
            True if recording was successful
        """
        print("Starting 3-second recording...")
        print("Recording will start in 2 seconds...")
        
        for i in range(2, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print("Recording now!")
        success = self.recorder.record_3_seconds(filename)
        
        if success:
            print(f"Recording completed and saved to: {filename}")
        else:
            print("Recording failed!")
            
        return bool(success)
        
    def stop_recording(self):
        """Stop current recording."""
        self.recorder.stop_recording()
        
    def cleanup(self):
        """Clean up resources."""
        self.recorder.cleanup()


# Example usage and testing
if __name__ == "__main__":
    # Test the simple recorder
    recorder = SimpleRecorder()
    
    # Record a test sound
    test_filename = "data/recordings/test_sound.wav"
    success = recorder.record_5_seconds(test_filename)
    
    if success:
        print("Test recording successful!")
        
        # Play it back
        player = AudioPlayer()
        player.play_file(test_filename)
        player.cleanup()
    
    recorder.cleanup() 