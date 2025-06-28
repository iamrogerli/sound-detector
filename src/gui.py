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
        
        ttk.Button(control_frame, text="Train Model", command=self.train_model).pack(pady=5)
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(pady=5)
        ttk.Button(control_frame, text="Test Model", command=self.test_model).pack(pady=5)
        
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
        # TODO: Implement model training
        self.log_text.insert('end', "Model training not implemented yet.\n")
        
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
        # TODO: Implement model testing
        self.log_text.insert('end', "Model testing not implemented yet.\n")
        
    def start_detection(self):
        """Start real-time sound detection."""
        self.detection_status_var.set("Detection running...")
        self.start_detection_button.config(state='disabled')
        self.stop_detection_button.config(state='normal')
        self.results_text.insert('end', "Detection started...\n")
        # TODO: Implement real-time detection
        
    def stop_detection(self):
        """Stop real-time sound detection."""
        self.detection_status_var.set("Detection stopped")
        self.start_detection_button.config(state='normal')
        self.stop_detection_button.config(state='disabled')
        self.results_text.insert('end', "Detection stopped.\n")
        
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def cleanup(self):
        """Clean up resources."""
        self.recording_manager.cleanup()
        self.simple_recorder.cleanup()


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