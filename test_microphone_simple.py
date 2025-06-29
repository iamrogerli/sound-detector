#!/usr/bin/env python3
"""
Simple microphone test to verify audio input is working.
"""

import pyaudio
import numpy as np
import time

def test_microphone():
    """Test microphone input and volume levels."""
    print("🎤 Simple Microphone Test")
    print("=" * 40)
    
    # Audio settings
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    
    audio = pyaudio.PyAudio()
    
    try:
        # Open stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("Microphone opened successfully!")
        print("Speak into your microphone for 5 seconds...")
        print("You should see volume levels changing as you speak.")
        print("-" * 40)
        
        # Monitor audio for 5 seconds
        start_time = time.time()
        max_volume = 0
        samples = 0
        
        while time.time() - start_time < 5:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.float32)
                
                # Calculate volume
                rms = np.sqrt(np.mean(audio_array**2))
                max_volume = max(max_volume, rms)
                samples += 1
                
                # Show volume bar
                volume_bar = "█" * int(rms * 50)
                print(f"\rVolume: {rms:.4f} {volume_bar:<50}", end="", flush=True)
                
            except Exception as e:
                print(f"\nError reading audio: {e}")
                break
        
        print(f"\n\nTest completed!")
        print(f"Maximum volume detected: {max_volume:.4f}")
        print(f"Average volume: {max_volume/2:.4f}")
        
        # Volume assessment
        if max_volume > 0.1:
            print("✅ Microphone is working well - good volume levels")
        elif max_volume > 0.01:
            print("⚠️  Microphone is working but volume is low")
            print("   Try speaking louder or moving closer to the microphone")
        else:
            print("❌ Microphone volume is very low")
            print("   Check microphone settings and permissions")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if max_volume < 0.05:
            print("• Increase microphone volume in system settings")
            print("• Move closer to the microphone")
            print("• Speak more loudly and clearly")
        else:
            print("• Your microphone should work fine for detection")
            print("• Try the real-time detection with confidence threshold 50-60%")
        
    except Exception as e:
        print(f"❌ Error opening microphone: {e}")
        print("Check microphone permissions and settings")
    
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        audio.terminate()

if __name__ == "__main__":
    test_microphone() 