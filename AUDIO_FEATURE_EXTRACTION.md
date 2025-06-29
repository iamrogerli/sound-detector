# Audio Feature Extraction in Sound Detection

## Overview

The sound detection system uses **MFCC (Mel-Frequency Cepstral Coefficients)** as the primary feature extraction method. This is a powerful technique widely used in speech recognition, music analysis, and audio classification.

## 🎵 What are MFCC Features?

### Mel-Frequency Cepstral Coefficients (MFCC)
MFCC features represent the **spectral characteristics** of audio in a way that mimics human auditory perception. They capture:

- **Frequency content** of the audio
- **Spectral envelope** information
- **Temporal characteristics** 
- **Human-like frequency perception** (Mel scale)

### Why MFCC?
1. **Human-like perception**: Mel scale approximates how humans hear frequencies
2. **Dimensionality reduction**: Compresses audio information efficiently
3. **Noise robustness**: Less sensitive to background noise
4. **Standard in audio ML**: Widely used in speech recognition and audio classification

## 🔧 Feature Extraction Process

### Step 1: Audio Loading
```python
# Load audio file using librosa
y, sr = librosa.load(audio_file, sr=None)
```
- **y**: Audio samples (time-domain)
- **sr**: Sample rate (typically 44.1 kHz)
- **sr=None**: Preserves original sample rate

### Step 2: MFCC Extraction
```python
# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```
- **n_mfcc=13**: Number of MFCC coefficients (standard choice)
- **Output**: 2D array (13 coefficients × time frames)

### Step 3: Feature Normalization
```python
# Normalize features
mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
```
- **Zero-mean normalization**: Centers data around 0
- **Unit variance**: Scales features to similar ranges
- **Improves model performance**: Prevents bias toward larger values

## 📊 Feature Structure

### Raw MFCC Features
```
Shape: (13, T) where T = number of time frames
- 13 MFCC coefficients per frame
- T frames based on audio duration
```

### Processed Features for Training
```python
# Flatten features
flattened_features = mfcc_features.flatten()

# Pad/truncate to fixed length
target_length = 13 * 100 = 1300 features
```

**Final feature vector**: 1300-dimensional vector per audio file

## 🧠 How MFCC Works

### 1. **Time-Domain to Frequency-Domain**
```
Audio samples → FFT → Power spectrum
```

### 2. **Mel Filter Bank**
```
Power spectrum → Mel filter bank → Mel spectrum
```
- **Mel scale**: Logarithmic frequency scale
- **Filter banks**: Overlapping triangular filters
- **Human-like**: Approximates cochlear response

### 3. **Logarithmic Compression**
```
Mel spectrum → log(mel_spectrum)
```
- **Dynamic range compression**: Reduces large variations
- **Perceptual relevance**: Matches human hearing

### 4. **Discrete Cosine Transform (DCT)**
```
log(mel_spectrum) → DCT → MFCC coefficients
```
- **Dimensionality reduction**: Keeps most important coefficients
- **Decorrelation**: Removes redundancy between features

## 📈 Feature Visualization

### MFCC Heatmap
```
Time →
┌─────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ MFCC 1
│ ░░░░████░░░░░░░░░░░░░░░░░░░░░░░░ │ MFCC 2  
│ ░░░░░░░░████░░░░░░░░░░░░░░░░░░░░ │ MFCC 3
│ ░░░░░░░░░░░░████░░░░░░░░░░░░░░░░ │ MFCC 4
│ ░░░░░░░░░░░░░░░░████░░░░░░░░░░░░ │ MFCC 5
│ ░░░░░░░░░░░░░░░░░░░░████░░░░░░░░ │ MFCC 6
│ ░░░░░░░░░░░░░░░░░░░░░░░░████░░░░ │ MFCC 7
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░████░ │ MFCC 8
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░███ │ MFCC 9
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█ │ MFCC 10
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ MFCC 11
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ MFCC 12
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ MFCC 13
└─────────────────────────────────┘
```

### What Each Coefficient Represents
- **MFCC 1**: Overall energy level
- **MFCC 2-4**: Spectral shape (formants)
- **MFCC 5-8**: Fine spectral details
- **MFCC 9-13**: Higher-order spectral characteristics

## 🎯 Feature Parameters

### Current Configuration
```python
AudioConfig:
- Sample Rate: 44,100 Hz
- Channels: 1 (mono)
- Format: Float32
- Chunk Size: 1024 samples

MFCC Parameters:
- n_mfcc: 13 coefficients
- Feature length: 1300 (13 × 100)
- Normalization: Zero-mean, unit variance
```

### Parameter Tuning Options
```python
# More detailed features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # More coefficients

# Different window sizes
mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=512)  # Smaller hop

# Additional features
mfccs_delta = librosa.feature.delta(mfccs)  # First derivative
mfccs_delta2 = librosa.feature.delta(mfccs, order=2)  # Second derivative
```

## 🔍 Feature Analysis Examples

### Voice vs. Doorbell Features
```
Voice (Human Speech):
- MFCC 1: High energy (0.8-1.2)
- MFCC 2-4: Formant structure (vowels)
- MFCC 5-8: Consonant characteristics
- Temporal: Continuous variation

Doorbell:
- MFCC 1: Sharp energy spike (1.5+)
- MFCC 2-4: Fixed frequency components
- MFCC 5-8: Harmonic structure
- Temporal: Sudden onset, decay
```

### Background Noise vs. Target Sound
```
Background Noise:
- MFCC 1: Low energy (0.1-0.3)
- MFCC 2-13: Random, unstructured
- Temporal: Consistent, low variation

Target Sound:
- MFCC 1: Higher energy (0.5+)
- MFCC 2-13: Structured patterns
- Temporal: Distinctive patterns
```

## 🛠️ Implementation Details

### Feature Extraction Pipeline
```python
def extract_features(audio_file: str, n_mfcc: int = 13) -> np.ndarray:
    # 1. Load audio
    y, sr = librosa.load(audio_file, sr=None)
    
    # 2. Extract MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 3. Normalize
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    return mfccs
```

### Training Data Preparation
```python
def prepare_training_data():
    features = []
    labels = []
    
    for audio_file in audio_files:
        # Extract features
        mfcc_features = extract_features(audio_file)
        
        # Flatten and pad
        flattened = mfcc_features.flatten()
        padded = pad_to_fixed_length(flattened, target_length=1300)
        
        features.append(padded)
        labels.append(get_sound_type(audio_file))
    
    return np.array(features), np.array(labels)
```

## 📊 Performance Considerations

### Computational Efficiency
- **MFCC extraction**: ~10-50ms per 3-second audio
- **Feature vector size**: 1300 dimensions
- **Memory usage**: ~5KB per audio file
- **Real-time capable**: Yes, for short audio segments

### Accuracy Factors
- **Audio quality**: Higher sample rate = better features
- **Duration**: Longer audio = more temporal information
- **Noise level**: Background noise affects feature quality
- **Feature normalization**: Critical for model performance

## 🔬 Advanced Feature Techniques

### 1. **Delta Features**
```python
# First and second derivatives
mfcc_delta = librosa.feature.delta(mfccs)
mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
```

### 2. **Spectral Features**
```python
# Spectral centroid, bandwidth, rolloff
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
```

### 3. **Temporal Features**
```python
# Zero crossing rate, RMS energy
zcr = librosa.feature.zero_crossing_rate(y)
rms = librosa.feature.rms(y=y)
```

### 4. **Combined Feature Sets**
```python
# Combine multiple feature types
combined_features = np.concatenate([
    mfccs.flatten(),
    spectral_centroid.flatten(),
    zcr.flatten(),
    rms.flatten()
])
```

## 🎯 Best Practices

### 1. **Audio Preprocessing**
- **Normalize volume**: Prevent bias toward loud sounds
- **Remove silence**: Focus on actual sound content
- **Consistent duration**: Pad/truncate to fixed length

### 2. **Feature Engineering**
- **Normalize features**: Zero-mean, unit variance
- **Feature selection**: Remove irrelevant coefficients
- **Dimensionality reduction**: PCA for high-dimensional features

### 3. **Model Considerations**
- **Feature scaling**: Important for SVM, neural networks
- **Feature importance**: Random Forest can show which MFCCs matter
- **Cross-validation**: Test feature robustness

## 🔍 Debugging Feature Extraction

### Common Issues
1. **Empty features**: Audio file corruption or format issues
2. **NaN values**: Division by zero in normalization
3. **Inconsistent shapes**: Variable audio durations
4. **Poor performance**: Inadequate feature normalization

### Debugging Tools
```python
# Check feature statistics
print(f"MFCC shape: {mfccs.shape}")
print(f"MFCC mean: {np.mean(mfccs):.4f}")
print(f"MFCC std: {np.std(mfccs):.4f}")
print(f"MFCC range: [{np.min(mfccs):.4f}, {np.max(mfccs):.4f}]")

# Visualize features
import matplotlib.pyplot as plt
plt.imshow(mfccs, aspect='auto', origin='lower')
plt.colorbar()
plt.title('MFCC Features')
plt.show()
```

## 📚 Further Reading

- **Librosa Documentation**: https://librosa.org/
- **MFCC Tutorial**: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
- **Audio Feature Extraction**: https://www.researchgate.net/publication/2614631
- **Machine Learning for Audio**: https://www.coursera.org/learn/audio-signal-processing

---

This feature extraction system provides a robust foundation for sound classification, balancing computational efficiency with feature richness for accurate sound detection. 