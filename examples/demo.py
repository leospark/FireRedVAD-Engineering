"""
FireRedVAD 流式推理示例

用法：
    python examples/demo.py audio.wav

功能：
    - 加载音频文件
    - 流式 VAD 检测
    - 输出语音片段时间戳
"""

import numpy as np
import soundfile as sf
from inference.streaming import StreamVAD


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法：python examples/demo.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # 加载音频
    audio, sample_rate = sf.read(audio_file)
    
    # 确保采样率为 16kHz
    if sample_rate != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples)
    
    # 初始化 VAD（使用带缓存的优化模型）
    vad = StreamVAD(model_path="../models/Stream-VAD.onnx")
    
    # 流式处理
    print("开始 VAD 检测...")
    segments = vad.process_audio(audio)
    
    # 输出结果
    print(f"\n检测到 {len(segments)} 个语音片段：")
    for i, (start, end, prob) in enumerate(segments, 1):
        print(f"  片段{i}: {start:.2f}s - {end:.2f}s (置信度：{prob:.2f})")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
