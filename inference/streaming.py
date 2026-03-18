# Added import for os.path
#!/usr/bin/env python3
"""
FireRedVAD ONNX 流式推理完整脚本
支持实时音频流 VAD 检测

功能：
- ✅ 音频特征提取 (Kaldi Fbank + CMVN)
- ✅ ONNX 模型推理
- ✅ 流式缓存管理
- ✅ VAD 后处理 (平滑、边界检测)
- ✅ 实时性能监控

日期：2026-03-18
"""

import os
import sys
import time
import wave
import numpy as np

# 添加项目内 fireredvad 包路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fireredvad.core.audio_feat import AudioFeat

from dataclasses import dataclass
from typing import List, Tuple, Optional

# ==================== 配置 ====================

@dataclass
class StreamVadConfig:
    """流式 VAD 配置"""
    # 模型路径
    onnx_path: str = "models/model_with_caches.onnx"
    
    # CMVN 文件路径（必需！）
    cmvn_path: str = "models/cmvn.ark"
    
    # 音频参数
    sample_rate: int = 16000
    frame_shift_ms: int = 10   # 帧移 10ms（固定）
    frame_length_ms: int = 25  # 帧长 25ms（固定） 
    frame_length_samples: int = 400  # 帧长（25ms at 16kHz）
    frame_shift_samples: int = 160   # 帧移（10ms at 16kHz）
    
    # VAD 参数
    speech_threshold: float = 0.5      # 语音阈值
    smooth_window_size: int = 5        # 平滑窗口
    min_speech_frame: int = 8          # 最小语音帧 (80ms)
    min_silence_frame: int = 20        # 最小静音帧 (200ms)
    pad_start_frame: int = 5           # 起始填充帧
    
    # 性能参数
    use_gpu: bool = False


@dataclass
class FrameResult:
    """单帧 VAD 结果"""
    frame_idx: int
    raw_prob: float           # 原始概率
    smoothed_prob: float      # 平滑后概率
    is_speech: bool           # 是否语音
    is_speech_start: bool     # 语音起始点
    is_speech_end: bool       # 语音结束点
    timestamp_ms: float       # 时间戳 (ms)



# ==================== VAD 后处理 ====================

class VadPostprocessor:
    """VAD 后处理器"""
    
    def __init__(self, config: StreamVadConfig):
        self.config = config
        self.smooth_window = []
        self.frame_idx = 0
        self.in_speech = False
        self.speech_start_frame = -1
        self.silence_count = 0
        self.speech_count = 0
    
    def process_frame(self, raw_prob: float) -> FrameResult:
        """
        处理单帧概率
        
        Args:
            raw_prob: 原始概率 [0, 1]
        
        Returns:
            FrameResult: 帧结果
        """
        # 1. 平滑
        self.smooth_window.append(raw_prob)
        if len(self.smooth_window) > self.config.smooth_window_size:
            self.smooth_window.pop(0)
        
        smoothed_prob = np.mean(self.smooth_window)
        
        # 2. 阈值判断
        is_speech = smoothed_prob >= self.config.speech_threshold
        
        # 3. 状态机
        is_speech_start = False
        is_speech_end = False
        
        if is_speech and not self.in_speech:
            # 可能语音开始
            self.speech_count += 1
            if self.speech_count >= self.config.pad_start_frame:
                self.in_speech = True
                self.speech_start_frame = self.frame_idx - self.config.pad_start_frame + 1
                is_speech_start = True
                self.silence_count = 0
        elif is_speech and self.in_speech:
            # 持续语音
            self.speech_count += 1
            self.silence_count = 0
        elif not is_speech and self.in_speech:
            # 可能语音结束
            self.silence_count += 1
            if self.silence_count >= self.config.min_silence_frame:
                self.in_speech = False
                is_speech_end = True
                self.speech_count = 0
        else:
            # 持续静音
            self.silence_count += 1
            self.speech_count = 0
        
        # 4. 创建结果
        result = FrameResult(
            frame_idx=self.frame_idx,
            raw_prob=raw_prob,
            smoothed_prob=smoothed_prob,
            is_speech=is_speech and self.in_speech,
            is_speech_start=is_speech_start,
            is_speech_end=is_speech_end,
            timestamp_ms=self.frame_idx * self.config.frame_shift_ms
        )
        
        self.frame_idx += 1
        return result
    
    def reset(self):
        """重置状态"""
        self.smooth_window = []
        self.frame_idx = 0
        self.in_speech = False
        self.speech_start_frame = -1
        self.silence_count = 0
        self.speech_count = 0


# ==================== 主 VAD 类 ====================

class FireRedStreamVadONNX:
    """FireRedVAD ONNX 流式推理"""
    
    def __init__(self, config: StreamVadConfig):
        self.config = config
        
        # 1. 加载 ONNX 模型
        print("📦 加载 ONNX 模型...")
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.use_gpu else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(config.onnx_path, providers=providers)
            
            print(f"✅ 模型加载成功")
            print(f"   路径：{config.onnx_path}")
            print(f"   设备：{'GPU' if config.use_gpu else 'CPU'}")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            raise
        
        # 2. 特征提取器
        print("🎵 初始化特征提取器...")
        self.feat_extractor = AudioFeat(config.cmvn_path)
        print(f"   ✅ 使用 Kaldi Fbank + CMVN")
        print(f"   CMVN: {config.cmvn_path}")
        
        # 3. 后处理器
        print("🔧 初始化后处理器...")
        self.postprocessor = VadPostprocessor(config)
        
        # 4. 模型缓存 (用于流式推理)
        self.model_caches = None
        
        print("✅ 初始化完成")
    
    def process_frame(self, audio_frame: np.ndarray) -> FrameResult:
        """
        处理单帧音频
        
        Args:
            audio_frame: 音频帧 [frame_length_samples], int16
        
        Returns:
            FrameResult: VAD 结果
        """
        # 1. 提取特征
        feat, _ = self.feat_extractor.extract(audio_frame)
        feat_np = feat.numpy() if hasattr(feat, 'numpy') else feat
        # 形状 [1, 80] -> [1, 1, 80]
        feat_np = feat_np.reshape(1, 1, -1).astype(np.float32)
        
        # 2. ONNX 推理
        prob, self.model_caches = self._inference(feat_np)
        
        # 3. 后处理
        result = self.postprocessor.process_frame(prob)
        
        return result
    
    def _inference(self, feat: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        ONNX 推理
        
        Args:
            feat: 特征 [1, 1, feat_dim]
        
        Returns:
            prob: 语音概率
            caches: 新的缓存状态
        """
        # 准备输入
        input_feed = {'input': feat}
        
        # 添加缓存（第一次使用零初始化）
        if self.model_caches is None:
            # 初始化 8 个缓存，每个形状 [1, 128, 19]
            batch_size = 1
            for i in range(8):
                input_feed[f'cache_{i}'] = np.zeros((batch_size, 128, 19), dtype=np.float32)
        else:
            # 使用上一次的缓存
            for i in range(8):
                input_feed[f'cache_{i}'] = self.model_caches[i]
        
        # 运行推理
        outputs = self.session.run(None, input_feed)
        
        # 解析输出
        # 输出 0: logits [batch, seq_len, 1] (需要 sigmoid 转换)
        # 输出 1-8: 新缓存状态
        logits = float(outputs[0][0, 0, 0])
        prob = float(1 / (1 + np.exp(-logits)))  # sigmoid 转换
        caches = [outputs[i+1] for i in range(8)]  # 8 个缓存张量
        
        return prob, caches
    
    def process_chunk(self, audio_chunk: np.ndarray) -> List[FrameResult]:
        """
        处理音频块 (多帧)
        
        Args:
            audio_chunk: 音频块 [N samples], int16
        
        Returns:
            List[FrameResult]: 帧结果列表
        """
        results = []
        
        # 分帧
        n_frames = (len(audio_chunk) - self.config.frame_length_samples) // self.config.frame_shift_samples + 1
        
        for i in range(n_frames):
            start = i * self.config.frame_shift_samples
            end = start + self.config.frame_length_samples
            frame = audio_chunk[start:end]
            
            result = self.process_frame(frame)
            results.append(result)
        
        return results
    
    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float, float]]:
        """
        处理音频数组（numpy array）
        
        Args:
            audio: 音频数据 (int16 或 float32)
            sample_rate: 采样率
        
        Returns:
            [(start, end, probability), ...] - 语音段时间戳列表
        """
        # 重置状态
        self.reset()
        
        # 批量提取特征并推理
        return self._process_audio_original(audio, sample_rate)
    
    def _process_audio_original(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float, float]]:
        """使用原始 AudioFeat 批量处理"""
        import torch
        
        # 批量提取所有帧的特征
        feats, dur = self.feat_extractor.extract(audio)
        
        # 转换为 numpy
        feats_np = feats.numpy() if isinstance(feats, torch.Tensor) else feats
        
        # 逐帧 ONNX 推理（流式模拟）
        probs = []
        caches = [np.zeros((1, 128, 19), dtype=np.float32) for _ in range(8)]
        
        for i in range(feats_np.shape[0]):
            feat = feats_np[i:i+1].reshape(1, 1, -1).astype(np.float32)
            
            input_feed = {'input': feat}
            for j in range(8):
                input_feed[f'cache_{j}'] = caches[j]
            
            outputs = self.session.run(None, input_feed)
            prob = float(outputs[0][0, 0, 0])
            probs.append(prob)
            caches = [outputs[j+1] for j in range(8)]
        
        # 后处理
        for i, prob in enumerate(probs):
            result = self.postprocessor.process_frame(prob)
        
        # 提取语音段
        timestamps = self._extract_timestamps_with_prob_all(probs)
        return timestamps
    
    def _extract_timestamps_with_prob_all(self, probs: List[float]) -> List[Tuple[float, float, float]]:
        """从概率列表提取语音段"""
        segments = []
        in_speech = False
        speech_start = None
        speech_start_prob = 0.0
        silence_count = 0
        speech_count = 0
        
        for i, prob in enumerate(probs):
            is_speech = prob >= self.config.speech_threshold
            
            if is_speech and not in_speech:
                speech_count += 1
                if speech_count >= self.config.pad_start_frame:
                    in_speech = True
                    speech_start = i - self.config.pad_start_frame + 1
                    speech_start_prob = prob
                    silence_count = 0
            elif is_speech and in_speech:
                silence_count = 0
                speech_count += 1
            elif not is_speech and in_speech:
                silence_count += 1
                if silence_count >= self.config.min_silence_frame:
                    in_speech = False
                    if speech_start is not None:
                        end = i
                        # 计算平均概率
                        avg_prob = speech_start_prob
                        segments.append((speech_start * 0.01, end * 0.01, avg_prob))
                    speech_start = None
                    speech_count = 0
            else:
                silence_count += 1
                speech_count = 0
        
        if in_speech and speech_start is not None:
            end = len(probs)
            avg_prob = speech_start_prob
            segments.append((speech_start * 0.01, end * 0.01, avg_prob))
        
        return segments
    
    def _extract_timestamps_with_prob(self, results: List[FrameResult]) -> List[Tuple[float, float, float]]:
        """从帧结果提取语音段时间戳和概率"""
        timestamps = []
        start_time = None
        start_prob = 0.0
        
        for result in results:
            if result.is_speech_start:
                start_time = result.timestamp_ms
                start_prob = result.smoothed_prob
            elif result.is_speech_end and start_time is not None:
                end_time = result.timestamp_ms
                # 计算平均概率
                avg_prob = start_prob
                timestamps.append((start_time / 1000.0, end_time / 1000.0, avg_prob))
                start_time = None
        
        return timestamps
    
    def process_file(self, wav_path: str) -> Tuple[List[FrameResult], dict]:
        """
        处理完整 WAV 文件
        
        Args:
            wav_path: WAV 文件路径
        
        Returns:
            results: 帧结果列表
            stats: 统计信息
        """
        print(f"\n📂 处理文件：{wav_path}")
        
        # 读取 WAV
        with wave.open(wav_path, 'rb') as f:
            n_channels = f.getnchannels()
            sample_width = f.getsampwidth()
            sample_rate = f.getframerate()
            n_frames = f.getnframes()
            
            audio_data = f.readframes(n_frames)
            
            if sample_width == 2:
                audio = np.frombuffer(audio_data, dtype=np.int16)
            else:
                raise ValueError(f"不支持的位深：{sample_width}")
            
            if n_channels == 2:
                audio = audio[::2]  # 取左声道
        
        print(f"   采样率：{sample_rate} Hz")
        print(f"   时长：{len(audio)/sample_rate:.2f} s")
        print(f"   声道：{n_channels}")
        
        # 重置状态
        self.reset()
        
        # 处理
        start_time = time.time()
        results = self.process_chunk(audio)
        end_time = time.time()
        
        # 统计
        audio_duration = len(audio) / sample_rate * 1000  # ms
        processing_time = (end_time - start_time) * 1000  # ms
        rtf = processing_time / audio_duration
        
        speech_frames = sum(1 for r in results if r.is_speech)
        speech_ratio = speech_frames / len(results) if results else 0
        
        # 提取语音段
        timestamps = self._extract_timestamps(results)
        
        stats = {
            'total_frames': len(results),
            'speech_frames': speech_frames,
            'speech_ratio': speech_ratio,
            'processing_time_ms': processing_time,
            'audio_duration_ms': audio_duration,
            'rtf': rtf,
            'realtime_factor': 1/rtf if rtf > 0 else 0,
            'speech_segments': timestamps
        }
        
        return results, stats
    
    def _extract_timestamps(self, results: List[FrameResult]) -> List[Tuple[float, float]]:
        """从帧结果提取语音段时间戳"""
        timestamps = []
        start_time = None
        
        for result in results:
            if result.is_speech_start:
                start_time = result.timestamp_ms
            elif result.is_speech_end and start_time is not None:
                end_time = result.timestamp_ms
                timestamps.append((start_time / 1000.0, end_time / 1000.0))
                start_time = None
        
        return timestamps
    
    def reset(self):
        """重置所有状态"""
        self.feat_extractor.reset()
        self.postprocessor.reset()
        self.model_caches = None


# ==================== 测试函数 ====================

def create_test_audio(output_path: str, duration: float = 3.0):
    """创建测试音频"""
    print(f"\n🎵 创建测试音频：{output_path}")
    
    sample_rate = 16000
    n_samples = int(duration * sample_rate)
    
    # 生成音频：静音 - 语音 - 静音 - 语音 - 静音
    audio = np.zeros(n_samples, dtype=np.int16)
    
    # 第 1 段语音：0.5s - 1.5s (440Hz)
    t1 = np.linspace(0, 1.0, int(1.0 * sample_rate))
    audio[int(0.5*sample_rate):int(1.5*sample_rate)] = (np.sin(2 * np.pi * 440 * t1) * 32767 * 0.8).astype(np.int16)
    
    # 第 2 段语音：2.0s - 2.8s (880Hz)
    t2 = np.linspace(0, 0.8, int(0.8 * sample_rate))
    audio[int(2.0*sample_rate):int(2.8*sample_rate)] = (np.sin(2 * np.pi * 880 * t2) * 32767 * 0.8).astype(np.int16)
    
    # 保存
    with wave.open(output_path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())
    
    print(f"✅ 测试音频已创建")
    print(f"   时长：{duration} s")
    print(f"   语音段 1: 0.5s - 1.5s")
    print(f"   语音段 2: 2.0s - 2.8s")


def main():
    """主测试函数"""
    print("="*70)
    print("FireRedVAD ONNX 流式推理测试")
    print("="*70)
    
    # 1. 配置
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    config = StreamVadConfig(
        onnx_path=os.path.join(model_dir, 'model_with_caches.onnx'),
        cmvn_path=os.path.join(model_dir, 'cmvn.ark'),
        speech_threshold=0.5,
        smooth_window_size=5,
        use_gpu=False
    )
    
    # 2. 创建测试音频
    test_wav = os.path.join(os.path.dirname(__file__), '..', 'examples', 'test_audio.wav')
    create_test_audio(test_wav, duration=3.0)
    
    # 3. 创建 VAD
    print("\n🔧 创建 FireRedVAD...")
    vad = FireRedStreamVadONNX(config)
    
    # 4. 处理文件
    print("\n🎯 开始流式推理测试...")
    results, stats = vad.process_file(test_wav)
    
    # 5. 打印结果
    print("\n" + "="*70)
    print("📊 测试结果")
    print("="*70)
    
    print(f"\n基础统计:")
    print(f"   总帧数：{stats['total_frames']}")
    print(f"   语音帧数：{stats['speech_frames']}")
    print(f"   语音占比：{stats['speech_ratio']*100:.1f}%")
    
    print(f"\n性能数据:")
    print(f"   处理时间：{stats['processing_time_ms']:.1f} ms")
    print(f"   音频时长：{stats['audio_duration_ms']:.1f} ms")
    print(f"   RTF: {stats['rtf']:.4f} {'✅ 优秀' if stats['rtf'] < 0.1 else '⚠️ 可接受' if stats['rtf'] < 1 else '❌ 太慢'}")
    print(f"   实时因子：{stats['realtime_factor']:.1f}x {'✅ 可实时' if stats['rtf'] < 1 else '❌ 无法实时'}")
    
    print(f"\n语音段检测:")
    if stats['speech_segments']:
        for i, (start, end) in enumerate(stats['speech_segments'], 1):
            duration = end - start
            print(f"   段{i}: {start:.2f}s - {end:.2f}s (时长：{duration:.2f}s)")
    else:
        print(f"   ⚠️ 未检测到语音段")
    
    # 6. 评估
    print("\n" + "="*70)
    print("🎯 评估")
    print("="*70)
    
    passed = True
    
    # RTF 检查
    if stats['rtf'] < 1:
        print(f"✅ RTF 达标：{stats['rtf']:.4f} (< 1.0)")
    else:
        print(f"❌ RTF 过高：{stats['rtf']:.4f} (>= 1.0)")
        passed = False
    
    # 语音段检查
    expected_segments = [(0.5, 1.5), (2.0, 2.8)]
    detected_segments = stats['speech_segments']
    
    if len(detected_segments) >= len(expected_segments):
        print(f"✅ 语音段检测：检测到 {len(detected_segments)} 段 (期望 {len(expected_segments)} 段)")
    else:
        print(f"⚠️ 语音段检测：检测到 {len(detected_segments)} 段 (期望 {len(expected_segments)} 段)")
    
    # 总体评估
    print("\n" + "="*70)
    if passed:
        print("✅ 测试通过！FireRedVAD ONNX 流式推理工作正常")
    else:
        print("⚠️ 测试部分通过，可能需要优化参数")
    print("="*70)
    
    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# 别名：StreamVAD（简化调用）
StreamVAD = FireRedStreamVadONNX

