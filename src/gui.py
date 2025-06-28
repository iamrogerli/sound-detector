"""
GUI module for the sound detection app.
Provides a simple interface for recording, training, and detection.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Optional
from .audio_recorder import RecordingManager, SimpleRecorder
import time


class SoundDetectionGUI:
    """Main GUI for the sound detection application."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sound Detection App")
        self.root.geometry("600x500")
        
        # Initialize components
        self.recording_manager = RecordingManager()
        self.simple_recorder = SimpleRecorder()
        self.current_recording_thread = None
        self.detector = None  # Will be initialized when needed
        
        # Create GUI elements
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.recording_frame = ttk.Frame(self.notebook)
        self.training_frame = ttk.Frame(self.notebook)
        self.detection_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.recording_frame, text="Recording")
        self.notebook.add(self.training_frame, text="Training")
        self.notebook.add(self.detection_frame, text="Detection")
        
        # Setup each tab
        self.setup_recording_tab()
        self.setup_training_tab()
        self.setup_detection_tab()
        
    def setup_recording_tab(self):
        """Setup the recording tab."""
        # Title
        title_label = ttk.Label(self.recording_frame, text="Audio Recording", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Sound type input
        type_frame = ttk.Frame(self.recording_frame)
        type_frame.pack(pady=10)
        
        ttk.Label(type_frame, text="Sound Type:").pack(side='left', padx=5)
        self.sound_type_var = tk.StringVar(value="test")
        self.sound_type_entry = ttk.Entry(type_frame, textvariable=self.sound_type_var, width=20)
        self.sound_type_entry.pack(side='left', padx=5)
        
        # Recording controls
        control_frame = ttk.Frame(self.recording_frame)
        control_frame.pack(pady=20)
        
        self.record_5_button = ttk.Button(control_frame, text="Record 5 Seconds", 
                                         command=self.start_recording_5)
        self.record_5_button.pack(side='left', padx=5)
        
        self.record_3_button = ttk.Button(control_frame, text="Record 3 Seconds", 
                                         command=self.start_recording_3)
        self.record_3_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Recording", 
                                     command=self.stop_recording, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Press and hold recording button
        hold_frame = ttk.Frame(self.recording_frame)
        hold_frame.pack(pady=10)
        
        ttk.Label(hold_frame, text="Press and Hold to Record:").pack(pady=5)
        
        self.hold_record_button = ttk.Button(hold_frame, text="🎤 Press to Start Recording", 
                                           style='Hold.TButton')
        self.hold_record_button.pack(pady=5)
        
        # Bind mouse events for press and hold
        self.hold_record_button.bind('<Button-1>', self.start_hold_recording)
        self.hold_record_button.bind('<ButtonRelease-1>', self.stop_hold_recording)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to record")
        status_label = ttk.Label(self.recording_frame, textvariable=self.status_var)
        status_label.pack(pady=10)
        
        # Recordings list
        list_frame = ttk.Frame(self.recording_frame)
        list_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Recent Recordings:").pack(anchor='w')
        
        # Create treeview for recordings
        columns = ('File', 'Type', 'Duration', 'Size')
        self.recordings_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.recordings_tree.heading(col, text=col)
            self.recordings_tree.column(col, width=100)
        
        self.recordings_tree.pack(fill='both', expand=True)
        
        # Buttons for recordings
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Play Selected", command=self.play_selected).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Delete Selected", command=self.delete_selected).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Refresh List", command=self.refresh_recordings).pack(side='left', padx=5)
        
        # Load initial recordings
        self.refresh_recordings()
        
    def setup_training_tab(self):
        """Setup the training tab."""
        # Title
        title_label = ttk.Label(self.training_frame, text="Model Training", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Training status
        self.training_status_var = tk.StringVar(value="No model trained yet")
        status_label = ttk.Label(self.training_frame, textvariable=self.training_status_var)
        status_label.pack(pady=10)
        
        # Training controls
        control_frame = ttk.Frame(self.training_frame)
        control_frame.pack(pady=20)
        
        self.train_button = ttk.Button(control_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)
        
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(pady=5)
        ttk.Button(control_frame, text="Test Model", command=self.test_model).pack(pady=5)
        ttk.Button(control_frame, text="Check Recordings", command=self.check_recordings).pack(pady=5)
        
        # Training progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.training_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.pack(fill='x', padx=20, pady=10)
        
        # Training log
        log_frame = ttk.Frame(self.training_frame)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(log_frame, text="Training Log:").pack(anchor='w')
        
        self.log_text = tk.Text(log_frame, height=10, wrap='word')
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Check recordings on startup
        self.check_recordings()
        
    def check_recordings(self):
        """Check if there are enough recordings to train a model."""
        try:
            from src.audio_utils import list_audio_files
            
            recordings = list_audio_files("data/recordings")
            
            if len(recordings) < 2:
                self.train_button.config(state='disabled')
                self.training_status_var.set("Need at least 2 recordings to train")
                self.log_text.insert('end', f"Found {len(recordings)} recordings. Need at least 2 to train a model.\n")
            else:
                # Count different sound types
                sound_types = set()
                for recording in recordings:
                    filename = os.path.basename(recording)
                    if '_' in filename:
                        sound_type = filename.split('_')[0]
                        sound_types.add(sound_type)
                
                if len(sound_types) < 2:
                    self.train_button.config(state='disabled')
                    self.training_status_var.set(f"Need at least 2 different sound types (found: {list(sound_types)})")
                    self.log_text.insert('end', f"Found {len(sound_types)} sound types: {list(sound_types)}. Need at least 2 different types.\n")
                else:
                    self.train_button.config(state='normal')
                    self.training_status_var.set(f"Ready to train: {len(recordings)} recordings, {len(sound_types)} types")
                    self.log_text.insert('end', f"✓ Ready to train! Found {len(recordings)} recordings with {len(sound_types)} sound types: {list(sound_types)}\n")
                    
        except Exception as e:
            self.log_text.insert('end', f"Error checking recordings: {e}\n")
            self.train_button.config(state='disabled')
        
    def setup_detection_tab(self):
        """Setup the detection tab."""
        # Title
        title_label = ttk.Label(self.detection_frame, text="Real-time Detection", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Detection status
        self.detection_status_var = tk.StringVar(value="Detection stopped")
        status_label = ttk.Label(self.detection_frame, textvariable=self.detection_status_var)
        status_label.pack(pady=10)
        
        # Detection controls
        control_frame = ttk.Frame(self.detection_frame)
        control_frame.pack(pady=20)
        
        self.start_detection_button = ttk.Button(control_frame, text="Start Detection", 
                                                command=self.start_detection)
        self.start_detection_button.pack(side='left', padx=5)
        
        self.stop_detection_button = ttk.Button(control_frame, text="Stop Detection", 
                                               command=self.stop_detection, state='disabled')
        self.stop_detection_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Test Detection", 
                  command=self.test_detection).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Clear Results", 
                  command=self.clear_detection_results).pack(side='left', padx=5)
        
        # Detection results
        results_frame = ttk.Frame(self.detection_frame)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(results_frame, text="Detection Results:").pack(anchor='w')
        
        self.results_text = tk.Text(results_frame, height=15, wrap='word')
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
        
    def start_recording_5(self):
        """Start 5-second recording."""
        self._start_recording(5)
        
    def start_recording_3(self):
        """Start 3-second recording."""
        self._start_recording(3)
        
    def _start_recording(self, duration: int):
        """Start recording with specified duration."""
        sound_type = self.sound_type_var.get().strip()
        if not sound_type:
            messagebox.showerror("Error", "Please enter a sound type")
            return
            
        self.status_var.set(f"Recording {duration} seconds...")
        self.record_5_button.config(state='disabled')
        self.record_3_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Start recording in a separate thread
        self.current_recording_thread = threading.Thread(
            target=self._record_audio,
            args=(sound_type, duration)
        )
        self.current_recording_thread.start()
        
    def _record_audio(self, sound_type: str, duration: int = 5):
        """Record audio in a separate thread."""
        try:
            if duration == 3:
                success = self.simple_recorder.record_3_seconds(
                    f"data/recordings/{sound_type}_{self._get_timestamp()}.wav"
                )
            else:
                success = self.simple_recorder.record_5_seconds(
                    f"data/recordings/{sound_type}_{self._get_timestamp()}.wav"
                )
            
            # Update GUI in main thread
            self.root.after(0, self._recording_finished, success)
        except Exception as e:
            self.root.after(0, self._recording_error, str(e))
            
    def _recording_finished(self, success: bool):
        """Handle recording completion."""
        self.status_var.set("Recording completed" if success else "Recording failed")
        self.record_5_button.config(state='normal')
        self.record_3_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.refresh_recordings()
        
        # Check if we can now train a model
        if success:
            self.check_recordings()
        
    def _recording_error(self, error: str):
        """Handle recording error."""
        self.status_var.set(f"Recording error: {error}")
        self.record_5_button.config(state='normal')
        self.record_3_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
    def stop_recording(self):
        """Stop current recording."""
        self.simple_recorder.stop_recording()
        self.status_var.set("Recording stopped")
        self.record_5_button.config(state='normal')
        self.record_3_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
    def start_hold_recording(self, event=None):
        """Start recording when hold button is pressed."""
        sound_type = self.sound_type_var.get().strip()
        if not sound_type:
            messagebox.showwarning("Warning", "Please enter a sound type")
            return
            
        # Disable other recording buttons
        self.record_5_button.config(state='disabled')
        self.record_3_button.config(state='disabled')
        self.hold_record_button.config(text="🎤 Recording... (Release to Stop)")
        
        # Start recording
        self.status_var.set("Recording... (press and hold)")
        self.simple_recorder.start_recording()
        
    def stop_hold_recording(self, event=None):
        """Stop recording when hold button is released."""
        if not self.simple_recorder.is_recording:
            return
            
        # Stop recording
        self.simple_recorder.stop_recording()
        
        # Get sound type and save recording
        sound_type = self.sound_type_var.get().strip()
        if sound_type:
            # Save the recording
            filename = f"{sound_type}_{self._get_timestamp()}.wav"
            filepath = os.path.join("data/recordings", filename)
            
            if self.simple_recorder.save_recording(filepath):
                self.status_var.set(f"Recording saved: {filename}")
                self.refresh_recordings()
                self.check_recordings()
            else:
                self.status_var.set("Failed to save recording")
        
        # Reset button states
        self.record_5_button.config(state='normal')
        self.record_3_button.config(state='normal')
        self.hold_record_button.config(text="🎤 Press to Start Recording")
        
    def play_selected(self):
        """Play selected recording."""
        selection = self.recordings_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a recording to play")
            return
            
        filepath = self.recordings_tree.item(selection[0])['values'][0]
        self.recording_manager.play_recording(filepath)
        
    def delete_selected(self):
        """Delete selected recording."""
        selection = self.recordings_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a recording to delete")
            return
            
        filepath = self.recordings_tree.item(selection[0])['values'][0]
        if messagebox.askyesno("Confirm", f"Delete recording: {filepath}?"):
            if self.recording_manager.delete_recording(filepath):
                self.refresh_recordings()
                
    def refresh_recordings(self):
        """Refresh the recordings list."""
        # Clear current items
        for item in self.recordings_tree.get_children():
            self.recordings_tree.delete(item)
            
        # Get recordings
        recordings = self.recording_manager.list_recordings()
        
        # Add to treeview
        for recording in recordings[-10:]:  # Show last 10 recordings
            info = self.recording_manager.get_recording_info(recording)
            filename = os.path.basename(recording)
            sound_type = filename.split('_')[0] if '_' in filename else 'unknown'
            
            self.recordings_tree.insert('', 'end', values=(
                recording,
                sound_type,
                f"{info['duration']:.1f}s",
                f"{info['size'] / 1024:.1f}KB"
            ))
            
    def train_model(self):
        """Train the sound recognition model."""
        self.log_text.insert('end', "Training model...\n")
        self.training_status_var.set("Training in progress...")
        
        # Disable training button during training
        self.train_button.config(state='disabled')
        
        # Start training in a separate thread
        training_thread = threading.Thread(target=self._train_model_thread)
        training_thread.start()
        
    def _train_model_thread(self):
        """Train model in a separate thread."""
        try:
            from src.model_trainer import SimpleTrainer
            
            # Update progress
            self.root.after(0, self._update_progress, 10, "Initializing trainer...")
            
            # Create trainer
            trainer = SimpleTrainer()
            
            # Update progress
            self.root.after(0, self._update_progress, 30, "Preparing training data...")
            
            # Train the model
            model_name = f"sound_model_{self._get_timestamp()}.pkl"
            
            # Update progress
            self.root.after(0, self._update_progress, 50, "Training model...")
            
            success = trainer.train_and_save(model_name)
            
            # Update progress
            self.root.after(0, self._update_progress, 80, "Saving model...")
            
            # Update GUI in main thread
            if success:
                self.root.after(0, self._training_success, model_name)
            else:
                self.root.after(0, self._training_failed)
                
        except Exception as e:
            self.root.after(0, self._training_error, str(e))
            
    def _update_progress(self, value: int, message: str):
        """Update progress bar and log message."""
        self.progress_var.set(value)
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')  # Auto-scroll to bottom
        
    def _training_success(self, model_name: str):
        """Handle successful training."""
        self.log_text.insert('end', f"✓ Model training completed successfully!\n")
        self.log_text.insert('end', f"Model saved as: {model_name}\n")
        self.training_status_var.set(f"Model trained: {model_name}")
        self.train_button.config(state='normal')
        
        # Update progress bar
        self.progress_var.set(100)
        
        # Test the model
        self.log_text.insert('end', "Testing model on sample recordings...\n")
        self._test_trained_model(model_name)
        
    def _training_failed(self):
        """Handle training failure."""
        self.log_text.insert('end', "✗ Model training failed!\n")
        self.log_text.insert('end', "Please ensure you have recorded sounds for different types.\n")
        self.training_status_var.set("Training failed")
        self.train_button.config(state='normal')
        self.progress_var.set(0)
        
    def _training_error(self, error: str):
        """Handle training error."""
        self.log_text.insert('end', f"✗ Training error: {error}\n")
        self.training_status_var.set("Training error")
        self.train_button.config(state='normal')
        self.progress_var.set(0)
        
    def _test_trained_model(self, model_name: str):
        """Test the trained model on sample recordings."""
        try:
            from src.model_trainer import SimpleTrainer
            from src.audio_utils import list_audio_files
            
            trainer = SimpleTrainer()
            model_path = f"data/models/{model_name}"
            
            # Get sample recordings
            recordings = list_audio_files("data/recordings")
            
            if recordings:
                # Test on first few recordings
                test_count = min(3, len(recordings))
                self.log_text.insert('end', f"Testing on {test_count} recordings:\n")
                
                for i, recording in enumerate(recordings[:test_count]):
                    try:
                        success = trainer.load_and_test(model_path, recording)
                        if success:
                            self.log_text.insert('end', f"  ✓ Test {i+1}: {os.path.basename(recording)}\n")
                        else:
                            self.log_text.insert('end', f"  ✗ Test {i+1}: {os.path.basename(recording)}\n")
                    except Exception as e:
                        self.log_text.insert('end', f"  ✗ Test {i+1} error: {e}\n")
                        
                self.log_text.insert('end', "Model testing completed!\n")
            else:
                self.log_text.insert('end', "No recordings found for testing.\n")
                
        except Exception as e:
            self.log_text.insert('end', f"Error testing model: {e}\n")
        
    def load_model(self):
        """Load a trained model."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.h5 *.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.training_status_var.set(f"Model loaded: {os.path.basename(filename)}")
            self.log_text.insert('end', f"Loaded model: {filename}\n")
            
    def test_model(self):
        """Test the trained model."""
        self.log_text.insert('end', "Testing model...\n")
        
        # Check if we have a model loaded or available
        model_files = [f for f in os.listdir("data/models") if f.endswith('.pkl')]
        
        if not model_files:
            self.log_text.insert('end', "No trained models found. Please train a model first.\n")
            return
            
        # Use the most recent model
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join("data/models", latest_model)
        
        self.log_text.insert('end', f"Testing model: {latest_model}\n")
        
        try:
            from src.model_trainer import SimpleTrainer
            from src.audio_utils import list_audio_files
            
            trainer = SimpleTrainer()
            
            # Get recordings to test on
            recordings = list_audio_files("data/recordings")
            
            if not recordings:
                self.log_text.insert('end', "No recordings found for testing.\n")
                return
                
            # Test on a few recordings
            test_count = min(5, len(recordings))
            self.log_text.insert('end', f"Testing on {test_count} recordings:\n")
            
            success_count = 0
            for i, recording in enumerate(recordings[:test_count]):
                try:
                    success = trainer.load_and_test(model_path, recording)
                    if success:
                        self.log_text.insert('end', f"  ✓ Test {i+1}: {os.path.basename(recording)}\n")
                        success_count += 1
                    else:
                        self.log_text.insert('end', f"  ✗ Test {i+1}: {os.path.basename(recording)}\n")
                except Exception as e:
                    self.log_text.insert('end', f"  ✗ Test {i+1} error: {e}\n")
                    
            accuracy = (success_count / test_count) * 100
            self.log_text.insert('end', f"Test completed! Accuracy: {accuracy:.1f}% ({success_count}/{test_count})\n")
            
        except Exception as e:
            self.log_text.insert('end', f"Error testing model: {e}\n")
        
    def start_detection(self):
        """Start real-time sound detection."""
        try:
            from src.sound_detector import RealTimeDetector
            
            # Check if we have a trained model
            model_files = [f for f in os.listdir("data/models") if f.endswith('.pkl')]
            
            if not model_files:
                messagebox.showerror("Error", "No trained models found. Please train a model first.")
                return
                
            # Use the most recent model
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join("data/models", latest_model)
            
            # Initialize detector
            self.detector = RealTimeDetector(model_path)
            
            # Set up detection callback
            def on_detection(detection_info):
                timestamp = time.strftime("%H:%M:%S", time.localtime(detection_info['timestamp']))
                result_text = f"[{timestamp}] DETECTED: {detection_info['sound_type']} (confidence: {detection_info['confidence']:.3f})\n"
                self.results_text.insert('end', result_text)
                self.results_text.see('end')  # Auto-scroll
                
            self.detector.on_detection = on_detection
            
            # Start detection
            success = self.detector.start_detection(
                target_sounds=None,  # Detect all sounds
                confidence_threshold=0.7,
                detection_interval=2.0
            )
            
            if success:
                self.detection_status_var.set(f"Detection running (model: {latest_model})")
                self.start_detection_button.config(state='disabled')
                self.stop_detection_button.config(state='normal')
                self.results_text.insert('end', f"Detection started with model: {latest_model}\n")
                self.results_text.insert('end', "Listening for sounds...\n")
            else:
                messagebox.showerror("Error", "Failed to start detection. Check if model is valid.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")
            self.detection_status_var.set("Detection error")
        
    def stop_detection(self):
        """Stop real-time sound detection."""
        if self.detector:
            self.detector.stop_detection()
            self.detector.cleanup()
            self.detector = None
            
        self.detection_status_var.set("Detection stopped")
        self.start_detection_button.config(state='normal')
        self.stop_detection_button.config(state='disabled')
        self.results_text.insert('end', "Detection stopped.\n")
        
    def test_detection(self):
        """Test detection on existing recordings without starting real-time monitoring."""
        try:
            from src.sound_detector import SimpleDetector
            from src.audio_utils import list_audio_files
            
            # Check if we have a trained model
            model_files = [f for f in os.listdir("data/models") if f.endswith('.pkl')]
            
            if not model_files:
                messagebox.showerror("Error", "No trained models found. Please train a model first.")
                return
                
            # Use the most recent model
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join("data/models", latest_model)
            
            # Initialize detector
            detector = SimpleDetector(model_path)
            
            # Get recordings to test on
            recordings = list_audio_files("data/recordings")
            
            if not recordings:
                self.results_text.insert('end', "No recordings found for testing.\n")
                return
                
            # Test on a few recordings
            test_count = min(5, len(recordings))
            self.results_text.insert('end', f"Testing detection on {test_count} recordings:\n")
            
            success_count = 0
            for i, recording in enumerate(recordings[:test_count]):
                try:
                    success = detector.test_detection(recording)
                    if success:
                        self.results_text.insert('end', f"  ✓ Test {i+1}: {os.path.basename(recording)}\n")
                        success_count += 1
                    else:
                        self.results_text.insert('end', f"  ✗ Test {i+1}: {os.path.basename(recording)}\n")
                except Exception as e:
                    self.results_text.insert('end', f"  ✗ Test {i+1} error: {e}\n")
                    
            accuracy = (success_count / test_count) * 100
            self.results_text.insert('end', f"Detection test completed! Accuracy: {accuracy:.1f}% ({success_count}/{test_count})\n")
            
            # Clean up
            detector.cleanup()
            
        except Exception as e:
            self.results_text.insert('end', f"Error testing detection: {e}\n")
        
    def clear_detection_results(self):
        """Clear detection results."""
        self.results_text.delete('1.0', 'end')
        
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def cleanup(self):
        """Clean up resources."""
        self.recording_manager.cleanup()
        self.simple_recorder.cleanup()
        
        # Clean up detector if running
        if self.detector:
            self.detector.stop_detection()
            self.detector.cleanup()


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = SoundDetectionGUI(root)
    
    # Handle window close
    def on_closing():
        app.cleanup()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main() 