# FireRedVAD-Engineering

**High-Precision Streaming Voice Activity Detection (VAD) Using ONNX Runtime.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ONNX](https://img.shields.io/badge/ONNX-1.10+-green.svg)](https://onnx.ai/)

---

## What is FireRedVAD?

FireRedVAD-Engineering is a production-ready, streaming Voice Activity Detection (VAD) system optimized for real-time applications. Built with ONNX runtime for maximum performance and minimal dependencies.

### Key Features

- ⚡ **Real-time Streaming** - Low-latency audio stream processing
- 🎯 **High Precision** - ONNX vs PyTorch max difference < 1e-5
- 📦 **Lightweight** - Only ONNX Runtime required (no PyTorch)
- 🔧 **Easy Integration** - Clean API for quick project integration
- 💻 **Cross-Platform** - Windows, Linux, macOS support

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/leospark/FireRedVAD-Engineering.git
cd FireRedVAD-Engineering

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from inference.streaming import StreamVAD
import soundfile as sf

# Initialize VAD
vad = StreamVAD(
    model_path="models/Stream-VAD.onnx",
    sample_rate=16000,
    threshold=0.5
)

# Load and process audio
audio, sr = sf.read("audio.wav", dtype='int16')
segments = vad.process_audio(audio)

# Output results
for start, end, prob in segments:
    print(f"Speech: {start:.2f}s - {end:.2f}s (confidence: {prob:.2f})")
```

### Run Examples

```bash
# Detect speech segments
python examples/demo.py examples/test_audio.wav

# Plot VAD probability curve
python examples/plot_vad_prob.py examples/test_audio.wav
```

---

## Project Structure

```
FireRedVAD-Engineering/
├── models/                     # Pre-trained models
│   ├── Stream-VAD.onnx        (95KB)    # Model structure
│   └── Stream-VAD.onnx.data   (2.2MB)   # Model weights
├── inference/                  # Core inference code
│   └── streaming.py           (19KB)    # Streaming VAD engine
├── examples/                   # Usage examples
│   ├── demo.py                # Basic detection demo
│   ├── plot_vad_prob.py       # Visualization tool
│   └── test_audio.wav         # Test audio file
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

---

## Technical Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sample Rate** | 16 kHz | Recommended input |
| **Frame Length** | 400 samples | 25 ms |
| **Frame Shift** | 160 samples | 10 ms |
| **Feature Dim** | 80 | Log-Mel + CMVN |
| **Model Size** | 2.3 MB | ONNX format |
| **Memory Usage** | < 50 MB | Runtime memory |
| **RTF (CPU)** | < 0.1 | Real-time factor |

---

## API Reference

### StreamVAD Class

```python
vad = StreamVAD(
    model_path="models/Stream-VAD.onnx",   # Model path
    sample_rate=16000,                      # Audio sample rate
    threshold=0.5,                          # Detection threshold (0-1)
    smooth_window_size=5,                   # Smoothing window
    min_speech_frame=8,                     # Min speech frames (80ms)
    min_silence_frame=20                    # Min silence frames (200ms)
)
```

### Methods

#### `process_audio(audio: np.ndarray) -> List[Tuple[float, float, float]]`

Process complete audio file.

**Input:** `audio` - numpy array (int16, mono)  
**Output:** `[(start, end, probability), ...]` - List of speech segments

#### `process_chunk(audio_chunk: np.ndarray) -> List[FrameResult]`

Stream process audio chunks for real-time applications.

**Input:** `audio_chunk` - Audio chunk (100-500ms recommended)  
**Output:** List of FrameResult objects

---

## Use Cases

### 1. ASR Preprocessing

```python
# Extract speech segments for ASR
segments = vad.process_audio(audio)
for start, end, _ in segments:
    speech = audio[int(start*16000):int(end*16000)]
    text = asr.recognize(speech)
```

### 2. Meeting Analysis

```python
# Detect speaker activity
segments = vad.process_audio(meeting_audio)
for start, end, prob in segments:
    print(f"Speaking: {start:.1f}s - {end:.1f}s")
```

### 3. Voice Activity Ratio

```python
# Calculate speech ratio
total = len(audio) / 16000
voice = sum(end - start for start, end, _ in segments)
print(f"Voice Activity: {voice/total:.2%}")
```

### 4. Real-time Detection

```python
import sounddevice as sd

def callback(chunk, frames, time, status):
    segments = vad.process_chunk(chunk.flatten())
    if segments:
        print("Speech detected!")

stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
```

---

## Performance

### Accuracy Validation

| Metric | Value |
|--------|-------|
| **ONNX vs PyTorch** | < 1e-5 max diff |
| **Precision** | High consistency |
| **Recall** | Optimized for speech |

### Inference Speed

| Device | RTF | Notes |
|--------|-----|-------|
| **CPU (Intel i7)** | 0.05-0.08 | Real-time ready |
| **GPU (NVIDIA)** | 0.02-0.03 | Faster inference |

---

## FAQ

**Q: How to process non-16kHz audio?**

```python
from scipy.signal import resample
audio_16k = resample(audio, int(len(audio) * 16000 / original_sr))
```

**Q: How to handle stereo audio?**

```python
if audio.ndim == 2:
    audio = audio.mean(axis=1)  # Convert to mono
```

**Q: Can I use this commercially?**

Yes! MIT License allows commercial use.

**Q: How to adjust sensitivity?**

```python
vad = StreamVAD(threshold=0.3)  # More sensitive
vad = StreamVAD(threshold=0.7)  # Stricter detection
```

---

## License

MIT License - Free for personal and commercial use. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Based on FireRedVAD model with engineering optimizations
- Thanks to the open source community

---

## Support

For issues and feature requests, please use [GitHub Issues](https://github.com/leospark/FireRedVAD-Engineering/issues).

**Enjoy FireRedVAD!** ⭐
