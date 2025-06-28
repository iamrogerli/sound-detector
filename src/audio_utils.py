"""
Audio utilities for sound detection app.
Handles recording, playback, and audio processing functions.
"""

import pyaudio
import wave
import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple, Optional
import threading
import time


class AudioConfig:
    """Audio configuration settings."""
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5


class AudioRecorder:
    """Handles audio recording functionality."""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        
    def start_recording(self) -> None:
        """Start recording audio."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.frames = []
        
        stream = self.audio.open(
            format=self.config.FORMAT,
            channels=self.config.CHANNELS,
            rate=self.config.RATE,
            input=True,
            frames_per_buffer=self.config.CHUNK
        )
        
        print("Recording started...")
        start_time = time.time()
        
        while self.is_recording and (time.time() - start_time) < self.config.RECORD_SECONDS:
            try:
                data = stream.read(self.config.CHUNK)
                self.frames.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        self.is_recording = False
        print("Recording stopped.")
        
    def stop_recording(self) -> None:
        """Stop recording audio."""
        self.is_recording = False
        
    def save_recording(self, filename: str) -> bool:
        """Save recorded audio to file."""
        if not self.frames:
            print("No audio data to save.")
            return False
            
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.config.FORMAT))
                wf.setframerate(self.config.RATE)
                wf.writeframes(b''.join(self.frames))
            print(f"Audio saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False
            
    def record_and_save(self, filename: str) -> bool:
        """Record for specified duration and save to file."""
        self.start_recording()
        return self.save_recording(filename)
        
    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()


class AudioPlayer:
    """Handles audio playback functionality."""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        
    def play_file(self, filename: str) -> None:
        """Play an audio file."""
        try:
            with wave.open(filename, 'rb') as wf:
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                    
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()


def extract_features(audio_file: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from audio file.
    
    Args:
        audio_file: Path to audio file
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MFCC features as numpy array
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Normalize features
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        return mfccs
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def get_audio_duration(filename: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        y, sr = librosa.load(filename, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0


def list_audio_files(directory: str) -> list:
    """List all audio files in directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(directory, file))
            
    return audio_files


def create_spectrogram(audio_file: str, save_path: Optional[str] = None) -> np.ndarray:
    """
    Create spectrogram from audio file.
    
    Args:
        audio_file: Path to audio file
        save_path: Optional path to save spectrogram image
        
    Returns:
        Spectrogram as numpy array
    """
    try:
        y, sr = librosa.load(audio_file, sr=None)
        
        # Create spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        return D
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None 