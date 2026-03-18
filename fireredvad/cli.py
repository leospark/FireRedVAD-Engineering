#!/usr/bin/env python3
"""
FireRedVAD Command-Line Interface

Usage:
    fireredvad audio.wav
    fireredvad audio.wav --output segments.json
    fireredvad audio.wav --plot vad_plot.png
    fireredvad --help
"""

import argparse
import json
import sys
import os

# Add parent directory to path (for development)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.streaming import StreamVAD, StreamVadConfig


def main():
    parser = argparse.ArgumentParser(
        description="FireRedVAD - Voice Activity Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fireredvad audio.wav
    fireredvad audio.wav --output segments.json
    fireredvad audio.wav --plot vad_plot.png
    fireredvad --threshold 0.3 audio.wav
        """
    )
    
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Input audio file (WAV format, 16kHz recommended)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for speech segments"
    )
    
    parser.add_argument(
        "--plot", "-p",
        help="Save VAD probability plot to file"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Speech detection threshold (0.0-1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--min-speech",
        type=float,
        default=0.08,
        help="Minimum speech duration in seconds (default: 0.08)"
    )
    
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.2,
        help="Minimum silence duration in seconds (default: 0.2)"
    )
    
    parser.add_argument(
        "--model",
        default="models/model_with_caches.onnx",
        help="Path to ONNX model (default: models/model_with_caches.onnx)"
    )
    
    parser.add_argument(
        "--cmvn",
        default="models/cmvn.ark",
        help="Path to CMVN file (default: models/cmvn.ark)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if audio file is provided
    if not args.audio_file:
        parser.print_help()
        sys.exit(1)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.cmvn):
        print(f"Error: CMVN file not found: {args.cmvn}", file=sys.stderr)
        sys.exit(1)
    
    # Load audio
    try:
        import soundfile as sf
        audio, sr = sf.read(args.audio_file, dtype='int16')
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize VAD
    if args.verbose:
        print(f"Loading model: {args.model}")
        print(f"Loading CMVN: {args.cmvn}")
    
    config = StreamVadConfig(
        onnx_path=args.model,
        cmvn_path=args.cmvn,
        speech_threshold=args.threshold,
        min_speech_frame=int(args.min_speech * 100),  # Convert to frames
        min_silence_frame=int(args.min_silence * 100),
    )
    
    vad = StreamVAD(config)
    
    # Process audio
    if args.verbose:
        print(f"Processing: {args.audio_file}")
        print(f"Duration: {len(audio)/sr:.2f}s")
    
    segments = vad.process_audio(audio, sample_rate=sr)
    
    # Output results
    print(f"\nDetected {len(segments)} speech segment(s):")
    for i, (start, end, prob) in enumerate(segments, 1):
        duration = end - start
        print(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s, confidence: {prob:.2f})")
    
    # Save to JSON
    if args.output:
        output_data = {
            "audio_file": args.audio_file,
            "duration": len(audio) / sr,
            "segments": [
                {
                    "start": start,
                    "end": end,
                    "probability": prob,
                    "duration": end - start
                }
                for start, end, prob in segments
            ]
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSegments saved to: {args.output}")
    
    # Generate plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Re-process to get frame-level probabilities
            vad.reset()
            probs = []
            times = []
            
            FRAME_LENGTH = 400
            FRAME_SHIFT = 160
            
            for i in range(0, len(audio) - FRAME_LENGTH + 1, FRAME_SHIFT):
                frame = audio[i:i+FRAME_LENGTH]
                result = vad.process_frame(frame)
                probs.append(result.raw_prob)
                times.append(i * FRAME_SHIFT / sr)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(times, probs, 'b-', linewidth=0.5, alpha=0.7, label='Speech Probability')
            ax.axhline(y=args.threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold {args.threshold}')
            ax.fill_between(times, probs, args.threshold,
                           where=(np.array(probs) >= args.threshold),
                           interpolate=True, alpha=0.3, color='green', label='Speech')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Speech Probability')
            ax.set_title(f'FireRedVAD - {os.path.basename(args.audio_file)}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1)
            
            plt.savefig(args.plot, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved to: {args.plot}")
            
        except ImportError:
            print("Warning: matplotlib not installed, skipping plot generation", file=sys.stderr)
        except Exception as e:
            print(f"Error generating plot: {e}", file=sys.stderr)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
