#!/usr/bin/env python3
"""
FireRedVAD - 绘制 VAD 概率曲线图

用法：
    python examples/plot_vad_prob.py audio.wav

功能：
    - 处理音频文件
    - 绘制 VAD 概率曲线
    - 保存结果图片（plot_vad_prob.png）
    - 保存概率数据（plot_vad_prob.npy）
"""

import sys
import os
import wave
import numpy as np
import onnxruntime as ort

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.streaming import StreamVAD, AudioFeatExtractor, StreamVadConfig


def process_audio_and_plot(wav_path, output_dir="."):
    """处理音频并绘制概率曲线"""
    
    print("="*80)
    print("FireRedVAD - VAD 概率曲线图")
    print("="*80)
    print()
    
    # 配置
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 400
    FRAME_SHIFT = 160
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载音频
    print(f"📂 加载音频：{wav_path}")
    with wave.open(wav_path, 'rb') as f:
        n_channels = f.getnchannels()
        sample_rate = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        
        if n_channels == 2:
            audio = audio[::2]
        
        # 重采样（如果需要）
        if sample_rate != SAMPLE_RATE:
            print(f"   ⚠️ 重采样：{sample_rate}Hz → {SAMPLE_RATE}Hz")
            num_samples = int(len(audio) * SAMPLE_RATE / sample_rate)
            indices = np.linspace(0, len(audio) - 1, num_samples)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)
    
    duration = len(audio) / SAMPLE_RATE
    print(f"   时长：{duration:.2f}s")
    print(f"   采样：{len(audio)} samples")
    print()
    
    # 初始化 VAD
    print("📦 加载模型...")
    config = StreamVadConfig()
    vad = StreamVAD(config)
    print("✅ 模型加载成功\n")
    
    # 流式推理
    print("🔍 流式推理中...")
    probs = []
    times = []
    
    import time
    start_time = time.time()
    
    for i in range(0, len(audio) - FRAME_LENGTH + 1, FRAME_SHIFT):
        frame = audio[i:i+FRAME_LENGTH].astype(np.float32) / 32768.0
        
        # 特征提取
        feat, _ = feat_extractor.extract(frame)
        if hasattr(feat, 'numpy'):
            feat_np = feat.numpy()
        elif hasattr(feat, 'detach'):
            feat_np = feat.detach().cpu().numpy()
        else:
            feat_np = feat
        
        # ONNX 推理
        feat_input = feat_np.reshape(1, 1, -1).astype(np.float32)
        input_feed = {'input': feat_input}
        for j, cache in enumerate(caches):
            input_feed[f'cache_{j}'] = cache
        
        outputs = session.run(None, input_feed)
        prob = float(outputs[0][0, 0, 0])
        caches = [outputs[j+1] for j in range(8)]
        
        probs.append(prob)
        times.append(i * FRAME_SHIFT / SAMPLE_RATE)
    
    processing_time = time.time() - start_time
    rtf = processing_time / duration
    
    print(f"✅ 推理完成")
    print(f"   帧数：{len(probs)}")
    print(f"   概率范围：[{min(probs):.6f}, {max(probs):.6f}]")
    print(f"   平均概率：{np.mean(probs):.6f}")
    print(f"   RTF: {rtf:.4f}")
    print()
    
    # 保存概率数据
    probs_path = os.path.join(output_dir, 'probs.npy')
    times_path = os.path.join(output_dir, 'times.npy')
    np.save(probs_path, np.array(probs))
    np.save(times_path, np.array(times))
    print(f"💾 概率数据已保存:")
    print(f"   {probs_path}")
    print(f"   {times_path}")
    print()
    
    # 绘图
    print("📈 绘制概率曲线图...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # 绘制概率曲线
        ax.plot(times, probs, 'b-', linewidth=0.5, alpha=0.7, label='语音概率')
        
        # 绘制阈值线
        ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='阈值 0.5')
        
        # 填充区域
        ax.fill_between(times, probs, 0.5, 
                       where=(np.array(probs) >= 0.5),
                       interpolate=True, alpha=0.3, color='green',
                       label='语音段')
        
        # 标签和标题
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('语音概率', fontsize=12)
        ax.set_title(f'FireRedVAD 语音检测概率曲线\n{os.path.basename(wav_path)} ({duration:.2f}s)', 
                    fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, times[-1])
        
        # 保存
        plot_path = os.path.join(output_dir, 'vad_prob_curve.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 图片已保存:")
        print(f"   {plot_path}")
        print()
        
        # 统计语音段
        speech_segments = []
        in_speech = False
        speech_start = None
        
        for i, (t, p) in enumerate(zip(times, probs)):
            if p >= 0.5 and not in_speech:
                in_speech = True
                speech_start = t
            elif p < 0.5 and in_speech:
                in_speech = False
                if speech_start is not None and (t - speech_start) > 0.1:
                    speech_segments.append((speech_start, t))
                speech_start = None
        
        print(f"📊 语音段统计:")
        print(f"   检测到 {len(speech_segments)} 个语音段")
        for i, (start, end) in enumerate(speech_segments[:10], 1):
            print(f"      {i}. {start:.2f}s - {end:.2f}s (时长：{end-start:.2f}s)")
        if len(speech_segments) > 10:
            print(f"      ... 还有 {len(speech_segments)-10} 个")
        print()
        
        return plot_path, probs_path, times_path, speech_segments
        
    except Exception as e:
        print(f"⚠️ 绘图失败：{e}")
        print("   已保存概率数据，可以用其他工具绘图")
        return None, probs_path, times_path, []


def main():
    # 测试音频
    test_files = [
        ('官方中文', ('test_audio.wav', os.path.join(os.path.dirname(__file__), 'test_audio.wav'))),
    ]
    
    output_base = os.path.join(os.path.dirname(__file__), '..', 'output')
    
    results = []
    
    for name, wav_path in test_files:
        if not os.path.exists(wav_path):
            print(f"⚠️ 文件不存在：{wav_path}\n")
            continue
        
        print(f"\n{'='*80}")
        print(f"处理：{name}")
        print(f"{'='*80}\n")
        
        output_dir = os.path.join(output_base, name)
        plot_path, probs_path, times_path, segments = process_audio_and_plot(wav_path, output_dir)
        
        results.append({
            'name': name,
            'wav_path': wav_path,
            'plot_path': plot_path,
            'probs_path': probs_path,
            'times_path': times_path,
            'segments': segments
        })
    
    # 总结
    print("\n" + "="*80)
    print("✅ 所有处理完成!")
    print("="*80)
    print()
    print("📁 输出文件:")
    for r in results:
        print(f"\n{r['name']}:")
        if r['plot_path']:
            print(f"   📊 图片：{r['plot_path']}")
        print(f"   💾 概率：{r['probs_path']}")
        print(f"   💾 时间：{r['times_path']}")
        print(f"   📍 语音段：{len(r['segments'])} 个")
    
    print()


if __name__ == "__main__":
    main()



