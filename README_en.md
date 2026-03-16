# 🔥 FireRedVAD-Open

**Lightweight Streaming Voice Activity Detection (VAD) Tool**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ONNX](https://img.shields.io/badge/ONNX-1.10+-green.svg)](https://onnx.ai/)

---

## 📖 Introduction

FireRedVAD-Open is a lightweight, high-performance streaming voice activity detection (VAD) tool based on ONNX inference engine.

**Key Features:**
- ⚡ **Real-time Streaming** - Low-latency audio stream detection
- 🎯 **High Precision** - Rigorously validated, ONNX vs PyTorch max difference < 1e-5
- 📦 **Lightweight** - Only requires ONNX Runtime, no PyTorch needed
- 🔧 **Easy Integration** - Clean API, quick integration into your projects
- 📱 **Low Resource** - Suitable for edge devices and real-time applications

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from inference.streaming import StreamVAD

# Initialize VAD
vad = StreamVAD()

# Process audio file
segments = vad.process_audio("audio.wav")

# Output speech segments
for start, end, prob in segments:
    print(f"Speech: {start:.2f}s - {end:.2f}s (confidence: {prob:.2f})")
```

### 3. Run Demo

```bash
python examples/demo.py your_audio_file.wav
```

---

## 📦 Project Structure

```
FireRedVAD-Open/
├── models/                     # Model files
│   ├── Stream-VAD.onnx        (95KB)    # Model structure
│   └── Stream-VAD.onnx.data   (2.2MB)   # Model weights (external data)
├── inference/                  # Inference code
│   └── streaming.py           (19KB)    # Streaming inference main file
├── examples/                   # Usage examples
│   └── demo.py                (1KB)     # Quick demo
├── requirements.txt            (182B)    # Python dependencies
├── LICENSE                     (1KB)     # MIT License
└── README.md                   (4.8KB)   # This file
```

---

## 🔧 Technical Specifications

### Audio Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sample Rate** | 16kHz | Recommended input sample rate |
| **Frame Length** | 400 samples | 25ms |
| **Frame Shift** | 160 samples | 10ms |
| **Feature Dimension** | 80 | Log-Mel spectrogram + CMVN |
| **Cache States** | 8 × (1, 128, 19) | LSTM/GRU caches |

### Model Information

| Model | Size | Description |
|-------|------|-------------|
| `Stream-VAD.onnx` | 95KB + 2.2MB | Streaming VAD model (with cache optimization, external data) |

---

## 📖 API Documentation

### StreamVAD Class

```python
from inference.streaming import StreamVAD

vad = StreamVAD(
    model_path="models/Stream-VAD.onnx",
    sample_rate=16000,
    threshold=0.5
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | - | ONNX model path |
| `sample_rate` | int | 16000 | Audio sample rate |
| `threshold` | float | 0.5 | VAD detection threshold |

#### Main Methods

**`process_audio(audio: np.ndarray) -> List[Tuple[float, float, float]]`**

Process complete audio file, return list of speech segments.

- **Input**: `audio` - Audio data (numpy array, mono)
- **Output**: `[(start, end, probability), ...]`
  - `start`: Segment start time (seconds)
  - `end`: Segment end time (seconds)
  - `probability`: Speech confidence (0-1)

**`process_chunk(audio_chunk: np.ndarray) -> List[Tuple[float, float, float]]`**

Stream processing of audio chunks, suitable for real-time applications.

- **Input**: `audio_chunk` - Audio chunk (recommend 100-500ms)
- **Output**: List of speech segments

---

## 💡 Use Cases

### 1. ASR Preprocessing

```python
# Extract speech segments, only process speech parts with ASR
segments = vad.process_audio(recording.wav)
for start, end, _ in segments:
    text = asr_engine.recognize(audio[start:end])
```

### 2. Meeting Transcription

```python
# Detect speaker activity periods
segments = vad.process_audio(meeting_recording.wav)
for start, end, prob in segments:
    print(f"Speech: {start:.1f}s - {end:.1f}s")
```

### 3. Audio Quality Assessment

```python
# Calculate Voice Activity Ratio
total_duration = len(audio) / 16000
voice_duration = sum(end - start for start, end, _ in segments)
var = voice_duration / total_duration
print(f"Voice Activity Ratio: {var:.2%}")
```

### 4. Real-time Voice Detection

```python
# Microphone real-time input
import sounddevice as sd

def audio_callback(chunk, frames, time, status):
    segments = vad.process_chunk(chunk)
    if segments:
        print("Speech detected!")

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
```

---

## 🔬 Performance Validation

### ONNX vs PyTorch Consistency

After rigorous validation, the maximum difference between ONNX model and original PyTorch model:

```
Max Difference: 0.0000002514 (< 1e-5) ✅
```

---

## 🛠️ Advanced Configuration

### Adjust Detection Threshold

```python
# Stricter (reduce false positives)
vad = StreamVAD(threshold=0.7)

# More sensitive (reduce false negatives)
vad = StreamVAD(threshold=0.3)
```

### Batch Processing

```python
# Process multiple audio files
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
for audio_file in audio_files:
    segments = vad.process_audio(audio_file)
    # Process results...
```

---

## 📝 FAQ

### Q: How to handle non-16kHz audio?

```python
from scipy.signal import resample

# Resample to 16kHz
num_samples = int(len(audio) * 16000 / original_sample_rate)
audio_16k = resample(audio, num_samples)
```

### Q: Can I use it for commercial projects?

A: Yes! This project uses MIT License, which allows commercial use.

---

## 📄 License

This project uses [MIT License](LICENSE) - free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- Engineered optimization based on FireRedVAD model
- Thanks to the open-source community for contributions

---

## 📬 Contact

- **GitHub Issues**: [Report issues or request features](https://github.com/leospark/FireRedVAD-Open/issues)
- **Email**: [To be filled]

---

## 🗺️ Roadmap

- [ ] Add batch processing optimization
- [ ] Support more audio formats
- [ ] Provide Docker image
- [ ] Add visualization interface
- [ ] Support multi-language speech detection

---

**🎉 Thank you for using FireRedVAD-Open!**
