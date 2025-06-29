#!/usr/bin/env python3
"""
Demo script showcasing the enhanced detection features:
- Real-time confidence display
- Live waveform visualization
- Color-coded confidence levels
"""

import sys
import os
import time
import threading

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_confidence_display():
    """Demo the confidence display features."""
    print("🎯 CONFIDENCE DISPLAY DEMO")
    print("=" * 40)
    
    # Simulate different confidence levels
    test_cases = [
        ("voice", 0.95, "HIGH"),
        ("doorbell", 0.78, "MEDIUM"), 
        ("background", 0.45, "LOW"),
        ("alarm", 0.88, "HIGH"),
        ("music", 0.62, "MEDIUM")
    ]
    
    for sound_type, confidence, level in test_cases:
        print(f"\n🎤 Detecting: {sound_type}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Level: {level}")
        
        # Simulate progress bar
        bar_length = 30
        filled_length = int(bar_length * confidence)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   Progress: [{bar}] {confidence:.1%}")
        
        time.sleep(0.5)  # Pause for effect
    
    print("\n✅ Confidence display demo completed!")

def demo_waveform_visualization():
    """Demo the waveform visualization features."""
    print("\n🌊 WAVEFORM VISUALIZATION DEMO")
    print("=" * 40)
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        print("Creating live waveform visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Live Audio Waveform Demo", fontsize=14, fontweight='bold')
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
        
        # Initialize data
        t = np.linspace(0, 3, 1000)
        line, = ax.plot(t, np.zeros(1000), 'b-', linewidth=2, label='Audio Input')
        
        # Animation function
        def animate(frame):
            # Generate realistic audio-like data
            time_val = frame * 0.05  # Time progression
            
            # Mix different frequencies to simulate real audio
            signal = (0.4 * np.sin(2 * np.pi * 440 * t + time_val) +  # 440 Hz (A note)
                     0.2 * np.sin(2 * np.pi * 880 * t + time_val * 2) +  # 880 Hz (A octave)
                     0.1 * np.sin(2 * np.pi * 220 * t + time_val * 0.5))  # 220 Hz (A lower)
            
            # Add some noise
            noise = np.random.normal(0, 0.05, 1000)
            waveform = signal + noise
            
            # Update line
            line.set_ydata(waveform)
            return line,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
        
        print("✅ Waveform animation created!")
        print("   - 20 FPS smooth animation")
        print("   - Realistic audio simulation")
        print("   - Multiple frequency components")
        print("   - Live amplitude visualization")
        
        plt.show()
        
    except ImportError:
        print("⚠️  Matplotlib not available - skipping waveform demo")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Waveform demo error: {e}")

def demo_detection_interface():
    """Demo the enhanced detection interface."""
    print("\n🎛️  ENHANCED DETECTION INTERFACE DEMO")
    print("=" * 40)
    
    print("Enhanced Detection Features:")
    print("┌─────────────────────────────────────────────────┐")
    print("│ 🎯 Current Detection: VOICE (HIGH)              │")
    print("│ Confidence: 85.0%                               │")
    print("│ ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░ │")
    print("└─────────────────────────────────────────────────┘")
    print()
    print("🌊 Live Audio Waveform:")
    print("   ┌─────────────────────────────────────────────┐")
    print("   │  ▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁ │")
    print("   │  ▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁ │")
    print("   │  ▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁ │")
    print("   └─────────────────────────────────────────────┘")
    print()
    print("📋 Detection History:")
    print("   [14:32:15] DETECTED: voice (confidence: 85.0%)")
    print("   [14:32:18] DETECTED: doorbell (confidence: 72.0%)")
    print("   [14:32:21] DETECTED: background (confidence: 45.0%)")
    print()
    print("🎮 Controls:")
    print("   [Start Detection] [Stop Detection] [Test Detection] [Clear Results]")

def main():
    """Run the enhanced features demo."""
    print("🎉 ENHANCED SOUND DETECTION FEATURES DEMO")
    print("=" * 50)
    print()
    print("This demo showcases the new enhanced features:")
    print("• Real-time confidence display with progress bar")
    print("• Color-coded confidence levels (HIGH/MEDIUM/LOW)")
    print("• Live waveform visualization")
    print("• Enhanced detection history with timestamps")
    print("• Smooth 20 FPS animation")
    print()
    
    # Run demos
    demo_confidence_display()
    demo_waveform_visualization()
    demo_detection_interface()
    
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETED!")
    print()
    print("To experience the full enhanced features:")
    print("1. Run: python main.py")
    print("2. Go to the Detection tab")
    print("3. Click 'Start Detection'")
    print("4. Watch the live confidence display and waveform!")
    print()
    print("Features you'll see:")
    print("✓ Real-time confidence percentage and progress bar")
    print("✓ Color-coded detection levels")
    print("✓ Live audio waveform animation")
    print("✓ Detailed detection history")
    print("✓ Smooth, responsive interface")

if __name__ == "__main__":
    main() 