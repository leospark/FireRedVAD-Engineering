# 🔥 FireRedVAD-Open

**轻量级流式语音活动检测（VAD）工具**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ONNX](https://img.shields.io/badge/ONNX-1.10+-green.svg)](https://onnx.ai/)
[![Stars](https://img.shields.io/github/stars/leospark/FireRedVAD-Open?style=social)](https://github.com/leospark/FireRedVAD-Open)

**语言:** [中文](README.md) | [English](README_en.md)

> ⭐ **如果喜欢这个项目，请给一个 Star！** 你的支持是我持续更新的动力！

---

## 📖 简介

FireRedVAD-Open 是一个轻量级、高性能的流式语音活动检测（VAD）工具，基于 ONNX 推理引擎实现。

**核心特点：**
- ⚡ **实时流式处理** - 支持低延迟音频流检测
- 🎯 **高精度** - 经过严格验证，ONNX vs PyTorch 最大差异 < 1e-5
- 📦 **轻量级** - 仅需 ONNX Runtime，无需 PyTorch
- 🔧 **易于集成** - 简洁的 API，快速接入你的项目
- 📱 **低资源占用** - 适合边缘设备和实时应用

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```python
from inference.streaming import StreamVAD

# 初始化 VAD
vad = StreamVAD()

# 处理音频文件
segments = vad.process_audio("audio.wav")

# 输出语音片段
for start, end, prob in segments:
    print(f"语音：{start:.2f}s - {end:.2f}s (置信度：{prob:.2f})")
```

### 3. 运行示例

```bash
# 基础示例：检测语音片段
python examples/demo.py your_audio_file.wav

# 画图示例：绘制 VAD 概率曲线（使用测试音频）
python examples/plot_vad_prob.py examples/test_audio.wav
```

**输出示例：**
- `plot_vad_prob.png` - VAD 概率曲线图
- `plot_vad_prob.npy` - 概率数据文件

---

## 📦 项目结构

```
FireRedVAD-Open/
├── models/                     # 模型文件
│   ├── Stream-VAD.onnx        (95KB)    # 模型结构
│   └── Stream-VAD.onnx.data   (2.2MB)   # 模型权重（外部数据）
├── inference/                  # 推理代码
│   └── streaming.py           (19KB)    # 流式推理主文件
├── examples/                   # 使用示例
│   ├── demo.py                (1KB)     # 快速演示
│   ├── plot_vad_prob.py       (8KB)     # 概率曲线绘图
│   └── test_audio.wav         (73KB)    # 测试音频
├── requirements.txt            (182B)    # Python 依赖
├── LICENSE                     (1KB)     # MIT 许可证
└── README.md                   (4.8KB)   # 本文件
```

---

## 🔧 技术规格

### 音频处理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **采样率** | 16kHz | 推荐输入采样率 |
| **帧长** | 400 样本 | 25ms |
| **帧移** | 160 样本 | 10ms |
| **特征维度** | 80 维 | Log-Mel 频谱 + CMVN |
| **缓存状态** | 8 个 (1, 128, 19) | LSTM/GRU 缓存 |

### 模型信息

| 模型 | 大小 | 说明 |
|------|------|------|
| `Stream-VAD.onnx` | 95KB + 2.2MB | 流式 VAD 模型（带缓存优化，外部数据） |

---

## 📖 API 文档

### StreamVAD 类

```python
from inference.streaming import StreamVAD

vad = StreamVAD(
    model_path="models/Stream-VAD.onnx",
    sample_rate=16000,
    threshold=0.5
)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | - | ONNX 模型路径 |
| `sample_rate` | int | 16000 | 音频采样率 |
| `threshold` | float | 0.5 | VAD 检测阈值 |

#### 主要方法

**`process_audio(audio: np.ndarray) -> List[Tuple[float, float, float]]`**

处理完整音频文件，返回语音片段列表。

- **输入**: `audio` - 音频数据 (numpy array, 单声道)
- **输出**: `[(start, end, probability), ...]`
  - `start`: 片段开始时间 (秒)
  - `end`: 片段结束时间 (秒)
  - `probability`: 语音置信度 (0-1)

**`process_chunk(audio_chunk: np.ndarray) -> List[Tuple[float, float, float]]`**

流式处理音频块，适合实时应用。

- **输入**: `audio_chunk` - 音频块 (建议 100-500ms)
- **输出**: 语音片段列表

---

## 💡 使用场景

### 1. 语音识别预处理

```python
# 提取语音片段，仅对语音部分进行 ASR
segments = vad.process_audio(recording.wav)
for start, end, _ in segments:
    text = asr_engine.recognize(audio[start:end])
```

### 2. 会议记录

```python
# 检测发言人活跃时间段
segments = vad.process_audio(meeting_recording.wav)
for start, end, prob in segments:
    print(f"发言：{start:.1f}s - {end:.1f}s")
```

### 3. 音频质量评估

```python
# 计算语音占比（Voice Activity Ratio）
total_duration = len(audio) / 16000
voice_duration = sum(end - start for start, end, _ in segments)
var = voice_duration / total_duration
print(f"语音占比：{var:.2%}")
```

### 4. 实时语音检测

```python
# 麦克风实时输入
import sounddevice as sd

def audio_callback(chunk, frames, time, status):
    segments = vad.process_chunk(chunk)
    if segments:
        print("检测到语音！")

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
```

### 5. 绘制 VAD 概率曲线

**使用测试音频：**
```bash
python examples/plot_vad_prob.py examples/test_audio.wav
```

**输出：**
- `plot_vad_prob.png` - 可视化概率曲线图
- `plot_vad_prob.npy` - 原始概率数据

**自定义音频：**
```bash
python examples/plot_vad_prob.py your_audio.wav
```

**用途：**
- 分析 VAD 检测效果
- 调整阈值参数
- 调试和验证

---

## 🔬 性能验证

### ONNX vs PyTorch 一致性

经过严格验证，ONNX 模型与原始 PyTorch 模型的最大差异：

```
最大差异：0.0000002514 (< 1e-5) ✅
```

---

## 🛠️ 高级配置

### 调整检测阈值

```python
# 更严格（减少误检）
vad = StreamVAD(threshold=0.7)

# 更敏感（减少漏检）
vad = StreamVAD(threshold=0.3)
```

### 批量处理

```python
# 处理多个音频文件
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
for audio_file in audio_files:
    segments = vad.process_audio(audio_file)
    # 处理结果...
```

---


## ⚙️ 配置说明

### 模型路径配置

默认情况下，模型路径使用相对路径：

`python
from inference.streaming import StreamVAD

# 使用默认路径（模型在项目目录内）
vad = StreamVAD()

# 或自定义路径
vad = StreamVAD(
    model_path="/path/to/your/Stream-VAD.onnx",
    cmvn_path="/path/to/your/cmvn.ark"  # 可选，None 使用默认值
)
`

### 目录结构

`
your_project/
├── your_code.py
└── FireRedVAD-Open/
    ├── models/
    │   └── Stream-VAD.onnx
    └── inference/
        └── streaming.py
`

`python
# 在你的项目中使用
import sys
sys.path.insert(0, 'FireRedVAD-Open')
from inference.streaming import StreamVAD

vad = StreamVAD(model_path='FireRedVAD-Open/models/Stream-VAD.onnx')
`

---

## 📝 常见问题

### Q: 如何处理非 16kHz 音频？

```python
from scipy.signal import resample

# 重采样到 16kHz
num_samples = int(len(audio) * 16000 / original_sample_rate)
audio_16k = resample(audio, num_samples)
```

### Q: 可以用于商业项目吗？

A: 可以！本项目采用 MIT 许可证，允许商业用途。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) - 允许自由使用、修改和分发。

---

## 🙏 致谢

- 基于 FireRedVAD 模型进行工程化优化
- 感谢开源社区的贡献

---

## 🗺️ 路线图

- [ ] 添加批量处理优化
- [ ] 支持更多音频格式
- [ ] 提供 Docker 镜像
- [ ] 添加可视化界面
- [ ] 支持多语言语音检测

---

**🎉 感谢使用 FireRedVAD-Open！**

> ⭐ **如果喜欢这个项目，请给一个 Star！**  
> 你的支持是我持续更新的动力！

**📄 许可：** MIT License - 允许自由使用、修改和分发

