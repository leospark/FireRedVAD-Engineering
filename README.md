# FireRedVAD-Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ONNX](https://img.shields.io/badge/ONNX-1.10+-green.svg)](https://onnx.ai/)
[![Validation](https://img.shields.io/badge/ONNX_vs_PyTorch-<1.2e--7-brightgreen)](https://github.com/leospark/FireRedVAD-Engineering)
[![Wheel](https://img.shields.io/badge/wheel-1.1.0-blue)](https://pypi.org/project/fireredvad-engineering/)

**Production-Ready Streaming Voice Activity Detection (VAD) with ONNX Runtime**

---

## 🔥 What's New

### v1.1.0 (2026-03-18)
- ✅ Package as Python wheel (`pip install fireredvad-engineering`)
- ✅ Add command-line interface (CLI) tool
- ✅ Include pre-built ONNX models and CMVN parameters
- ✅ Verified ONNX vs PyTorch consistency (max diff < 1.19×10⁻⁷)
- ✅ All 230 test frames validated
- ✅ Project is fully self-contained (includes fireredvad.core)

---

## 📖 What is FireRedVAD?

FireRedVAD-Engineering is a **production-ready**, streaming Voice Activity Detection (VAD) system optimized for real-time applications. Built with ONNX runtime for maximum performance and cross-platform compatibility.

Based on the [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) model by FireRed Team (Xiaohongshu).

### Key Features

- ⚡ **Real-time Streaming** - Low-latency audio stream processing (RTF < 0.1)
- 🎯 **High Precision** - ONNX vs PyTorch max difference **< 1.19×10⁻⁷** (validated)
- 📦 **Lightweight** - Only ONNX Runtime + Kaldi Fbank required (~2.3MB)
- 🔧 **Easy Integration** - Clean API and CLI for quick integration
- 💻 **Cross-Platform** - Windows, Linux, macOS support
- 🎤 **Multiple Tasks** - VAD (Voice Activity Detection) + AED (Audio Event Detection)

---

## 🚀 Quick Start

### Installation

#### Option 1: Install from PyPI (Recommended)

```bash
pip install fireredvad-engineering
```

#### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/leospark/FireRedVAD-Engineering.git
cd FireRedVAD-Engineering

# Install dependencies
pip install -r requirements.txt
```

---

### Usage

#### Option 1: Python API

```python
from fireredvad.inference.streaming import StreamVAD, StreamVadConfig
import soundfile as sf

# Initialize VAD
config = StreamVadConfig(
    onnx_path="models/model_with_caches.onnx",
    cmvn_path="models/cmvn.ark",  # Required!
    speech_threshold=0.5
)
vad = StreamVAD(config)

# Process audio file
audio, sr = sf.read("audio.wav", dtype='int16')
segments = vad.process_audio(audio, sample_rate=sr)

# Output results
print(f"Detected {len(segments)} speech segments:")
for start, end, prob in segments:
    print(f"  {start:.2f}s - {end:.2f}s (confidence: {prob:.2f})")
```

#### Option 2: Command-Line Interface (CLI)

```bash
# Basic usage
fireredvad audio.wav

# Save results to JSON
fireredvad audio.wav --output segments.json

# Generate probability plot
fireredvad audio.wav --plot vad_plot.png

# Adjust sensitivity (lower threshold = more sensitive)
fireredvad audio.wav --threshold 0.3

# Verbose output
fireredvad audio.wav --verbose
```

---

## 📁 Project Structure

```
FireRedVAD-Engineering/
├── models/                          # Pre-trained models
│   ├── model_with_caches.onnx       (96KB)   # ONNX model structure
│   ├── model_with_caches.onnx.data  (2.2MB)  # Model weights
│   └── cmvn.ark                     (1.3KB)  # ⚠️ Required! CMVN parameters
├── fireredvad/                      # Python package
│   ├── core/
│   │   ├── audio_feat.py            # Kaldi Fbank + CMVN feature extraction
│   │   └── detect_model.py          # PyTorch model (for validation)
│   ├── cli.py                       # Command-line interface
│   └── __init__.py
├── inference/                       # Inference engine
│   └── streaming.py                 # Streaming VAD engine
├── examples/                        # Usage examples
│   ├── demo.py                      # Basic detection demo
│   └── plot_vad_prob.py             # Visualization tool
├── tests/                           # Test scripts
│   ├── verify_onnx_vs_pytorch.py    # ONNX vs PyTorch validation
│   └── plot_comparison.py           # Generate comparison plots
├── setup.py                         # Package setup
├── pyproject.toml                   # Modern Python package config
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 📊 Technical Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sample Rate** | 16 kHz | Required input sample rate |
| **Frame Length** | 400 samples | 25 ms |
| **Frame Shift** | 160 samples | 10 ms |
| **Feature Dim** | 80 | Log-Mel Fbank + CMVN |
| **Model Size** | 2.3 MB | ONNX format |
| **Memory Usage** | < 50 MB | Runtime memory |
| **RTF (CPU)** | < 0.1 | Real-time factor |
| **Cache States** | 8 × [1, 128, 19] | Streaming cache tensors |

---

## 🔬 Validation Results

### ONNX vs PyTorch Consistency

**Test Audio:** Real Chinese speech (2.32s, 230 frames)

| Metric | Value |
|--------|-------|
| **Total Frames Tested** | 230 |
| **Max Difference** | 1.19×10⁻⁷ |
| **Mean Difference** | 3.29×10⁻⁸ |
| **Median Difference** | 2.61×10⁻⁸ |
| **< 1e-7** | 93.9% of frames ✅ |
| **< 1e-5** | 100% of frames ✅ |

**Conclusion:** ✅ **All 230 frames have difference < 1e-5** (requirement threshold)

### Speech Detection Test

| Test Audio | Duration | Detected Segments | Time Range | Confidence |
|------------|----------|-------------------|------------|------------|
| test_audio.wav | 2.32s | 1 segment | 0.49s - 1.80s | 0.99 |

---

## 🔧 API Reference

### StreamVadConfig

```python
config = StreamVadConfig(
    onnx_path="models/model_with_caches.onnx",    # ONNX model path (required)
    cmvn_path="models/cmvn.ark",                   # CMVN parameters (required)
    sample_rate=16000,                             # Audio sample rate
    frame_shift_ms=10,                             # Frame shift in ms
    speech_threshold=0.5,                          # Detection threshold (0-1)
    smooth_window_size=5,                          # Smoothing window size
    min_speech_frame=8,                            # Min speech frames (80ms)
    min_silence_frame=20,                          # Min silence frames (200ms)
    pad_start_frame=5,                             # Padding frames at start
    use_gpu=False                                  # Use GPU acceleration
)
```

### StreamVAD Class

```python
vad = StreamVAD(config)

# Process complete audio
segments = vad.process_audio(audio, sample_rate=16000)
# Returns: [(start, end, probability), ...]

# Process single frame (for streaming)
result = vad.process_frame(audio_frame)
# Returns: FrameResult object

# Process audio chunk (for streaming)
results = vad.process_chunk(audio_chunk)
# Returns: [FrameResult, ...]

# Reset all states (for new audio stream)
vad.reset()
```

### CLI Options

```bash
fireredvad audio.wav [OPTIONS]

Options:
  -o, --output PATH        Output JSON file for speech segments
  -p, --plot PATH          Save VAD probability plot to file
  -t, --threshold FLOAT    Speech detection threshold (0.0-1.0, default: 0.5)
  --min-speech FLOAT       Minimum speech duration in seconds (default: 0.08)
  --min-silence FLOAT      Minimum silence duration in seconds (default: 0.2)
  --model PATH             Path to ONNX model
  --cmvn PATH              Path to CMVN file
  -v, --verbose            Verbose output
  --help                   Show help message
```

---

## 💡 Use Cases

### 1. ASR Preprocessing

```python
segments = vad.process_audio(audio, sample_rate=16000)
for start, end, _ in segments:
    speech = audio[int(start*16000):int(end*16000)]
    text = asr.recognize(speech)
```

### 2. Meeting Analysis

```python
segments = vad.process_audio(meeting_audio, sample_rate=16000)
for start, end, prob in segments:
    print(f"Speaking: {start:.1f}s - {end:.1f}s (conf: {prob:.2f})")
```

### 3. Voice Activity Ratio

```python
total_duration = len(audio) / 16000
voice_duration = sum(end - start for start, end, _ in segments)
print(f"Voice Activity: {voice_duration/total_duration:.2%}")
```

### 4. Real-time Streaming

```python
import sounddevice as sd

vad = StreamVAD(config)
vad.reset()

def audio_callback(chunk, frames, time, status):
    results = vad.process_chunk(chunk.flatten())
    if any(r.is_speech for r in results):
        print("Speech detected!")

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
```

---

## ⚙️ Dependencies

### Core Requirements

```txt
onnxruntime>=1.10.0      # ONNX inference
numpy>=1.20.0            # Numerical operations
soundfile>=0.10.0        # Audio file I/O
kaldiio>=2.18.0          # Kaldi CMVN loading
kaldi-native-fbank>=1.7.0 # Kaldi Fbank feature extraction
```

### Optional (for development)

```txt
torch>=1.10.0            # PyTorch (for validation)
matplotlib>=3.5.0        # Visualization
scipy>=1.7.0             # Audio processing
```

---

## ❓ FAQ

### Q: Why is `cmvn.ark` required?

**A:** CMVN (Cepstral Mean and Variance Normalization) is critical for feature normalization. Without it, the model cannot properly process audio features and will output incorrect probabilities.

### Q: How to process non-16kHz audio?

```python
from scipy.signal import resample
audio_16k = resample(audio, int(len(audio) * 16000 / original_sr))
```

### Q: How to handle stereo audio?

```python
if audio.ndim == 2:
    audio = audio.mean(axis=1)  # Convert to mono
```

### Q: Can I use this commercially?

**A:** Yes! MIT License allows commercial use.

### Q: How to adjust sensitivity?

```python
# More sensitive (detect quieter speech)
config = StreamVadConfig(speech_threshold=0.3)

# Stricter (fewer false positives)
config = StreamVadConfig(speech_threshold=0.7)
```

### Q: What's the difference between `model_with_caches.onnx` and `model.onnx`?

- `model_with_caches.onnx`: Optimized for streaming with explicit cache states (recommended)
- `model.onnx`: Original model without cache optimization

---

## 🗺️ Roadmap

### v1.1.0 (Current)
- ✅ Package as Python wheel
- ✅ Add CLI tool
- ✅ Include pre-built models
- ✅ Improve documentation

### Future Plans
- [ ] Optimize for mobile deployment (TensorFlow Lite, CoreML)

---

## 📝 License

MIT License - Free for personal and commercial use. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Based on [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) model
- Original authors: Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu (Xiaohongshu)
- Thanks to the open source community

---

**Last Updated:** 2026-03-18  
**Version:** 1.1.0  
**Status:** ✅ Production Ready

**Enjoy FireRedVAD!** ⭐
