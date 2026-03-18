"""
FireRedVAD 流式推理示例

用法：
    python examples/demo.py audio.wav

功能：
    - 加载音频文件
    - 流式 VAD 检测
    - 输出语音片段时间戳
"""

import os
import sys
import numpy as np
import soundfile as sf

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.streaming import StreamVAD


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法：python examples/demo.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # 加载音频（int16 格式）
    audio, sample_rate = sf.read(audio_file, dtype='int16')
    
    # 确保采样率为 16kHz
    if sample_rate != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples).astype(np.int16)
        sample_rate = 16000
    
    # 初始化 VAD（使用带缓存的优化模型）
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inference.streaming import StreamVadConfig
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'model_with_caches.onnx')
    cmvn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'cmvn.ark')
    config = StreamVadConfig(onnx_path=model_path, cmvn_path=cmvn_path)
    vad = StreamVAD(config)
    
    # 流式处理
    print("开始 VAD 检测...")
    segments = vad.process_audio(audio, sample_rate=sample_rate)
    
    # 输出结果
    print(f"\n检测到 {len(segments)} 个语音片段：")
    for i, (start, end, prob) in enumerate(segments, 1):
        print(f"  片段{i}: {start:.2f}s - {end:.2f}s (置信度：{prob:.2f})")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
