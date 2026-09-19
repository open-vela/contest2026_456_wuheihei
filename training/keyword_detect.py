#!/usr/bin/env python3
"""
Keyword Detection System — MFCC + DTW Template Matching

Recognizes 2-3 custom voice keywords locally:
  - "救命" (help/emergency)
  - "帮助" (help/assist)
  - "报警" (alarm/report)

Algorithm:
  1. Extract MFCC features (13 coeffs × N frames) from input audio
  2. Pre-compute MFCC templates for each keyword (multiple samples each)
  3. Match input against each template using DTW (Dynamic Time Warping)
  4. Return keyword with lowest DTW distance (if below threshold)

This is a lightweight, training-free approach suitable for embedded devices.
No external ML framework needed — pure numpy implementation.

Usage:
  # Generate synthetic keyword templates
  python keyword_detect.py --generate_templates

  # Detect keyword from WAV file
  python keyword_detect.py --detect data/keywords/test_help.wav

  # Run full evaluation
  python keyword_detect.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from audio_utils import read_wav_mono, write_wav_mono

# ---------------------------------------------------------------------------
# MFCC Feature Extraction (pure numpy, no external DSP library)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_LEN = 400      # 25ms at 16kHz
HOP_LEN = 160        # 10ms at 16kHz
NUM_MFCC = 13        # 13 MFCC coefficients
NUM_FILTERS = 26     # Mel filter bank size
NFFT = 512


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_filters: int, nfft: int, sample_rate: int) -> np.ndarray:
    """Pre-compute Mel filter bank matrix."""
    high_freq = sample_rate / 2
    mel_min = _hz_to_mel(0)
    mel_max = _hz_to_mel(high_freq)
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, nfft // 2 + 1), dtype=np.float32)
    for m in range(1, num_filters + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        for k in range(left, center):
            if center > left:
                fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank[m - 1, k] = (right - k) / (right - center)
    return fbank


# DCT-II matrix for MFCC computation
def _dct_matrix(num_mfcc: int, num_filters: int) -> np.ndarray:
    """Pre-compute DCT-II matrix."""
    matrix = np.zeros((num_mfcc, num_filters), dtype=np.float32)
    for k in range(num_mfcc):
        for n in range(num_filters):
            matrix[k, n] = math.cos(math.pi * k * (2 * n + 1) / (2 * num_filters))
    matrix *= math.sqrt(2.0 / num_filters)
    matrix[0, :] *= math.sqrt(0.5)  # Scaling for k=0
    return matrix


# Pre-compute tables
_MEL_FB = _mel_filterbank(NUM_FILTERS, NFFT, SAMPLE_RATE)
_DCT_MAT = _dct_matrix(NUM_MFCC, NUM_FILTERS)


def extract_mfcc(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features from PCM audio.

    Args:
        pcm: int16 mono audio samples
        sample_rate: sample rate (must be 16000)

    Returns:
        MFCC matrix of shape (num_frames, NUM_MFCC)
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sample_rate}Hz")

    # Normalize to float
    x = pcm.astype(np.float32) / 32768.0

    # Pre-emphasis filter (high-pass): y[n] = x[n] - 0.97 * x[n-1]
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])

    # Frame the signal
    num_frames = max(1, 1 + (len(x) - FRAME_LEN) // HOP_LEN)
    frames = np.zeros((num_frames, FRAME_LEN), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_LEN
        end = min(start + FRAME_LEN, len(x))
        frames[i, :end - start] = x[start:end]

    # Apply Hamming window
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))
    frames *= hamming

    # FFT (magnitude spectrum)
    fft_result = np.fft.rfft(frames, n=NFFT, axis=1)
    power_spectrum = np.abs(fft_result) ** 2

    # Mel filter bank
    mel_energy = power_spectrum @ _MEL_FB.T  # (num_frames, NUM_FILTERS)
    mel_energy = np.where(mel_energy < 1e-10, 1e-10, mel_energy)
    log_mel = np.log(mel_energy)

    # DCT to get MFCC
    mfcc = log_mel @ _DCT_MAT.T  # (num_frames, NUM_MFCC)

    return mfcc


# ---------------------------------------------------------------------------
# DTW (Dynamic Time Warping) — pure numpy
# ---------------------------------------------------------------------------

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute DTW distance between two MFCC sequences.

    Uses standard DTW with Euclidean local distance.
    Sakoe-Chiba band constraint for efficiency.

    Args:
        seq1: MFCC matrix (T1, D)
        seq2: MFCC matrix (T2, D)

    Returns:
        Normalized DTW distance
    """
    t1, d = seq1.shape
    t2, _ = seq2.shape

    if t1 == 0 or t2 == 0:
        return float("inf")

    # Sakoe-Chiba band width (allow 30% time warping)
    band = max(abs(t1 - t2), int(0.3 * max(t1, t2)))

    # Initialize cost matrix
    INF = float("inf")
    cost = np.full((t1 + 1, t2 + 1), INF, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, t1 + 1):
        j_start = max(1, i - band)
        j_end = min(t2, i + band)
        for j in range(j_start, j_end + 1):
            # Euclidean distance between frames
            dist = np.sqrt(np.sum((seq1[i - 1] - seq2[j - 1]) ** 2))
            cost[i, j] = dist + min(cost[i - 1, j],      # insertion
                                    cost[i, j - 1],      # deletion
                                    cost[i - 1, j - 1])  # match

    # Normalize by path length
    return float(cost[t1, t2]) / max(t1, t2)


# ---------------------------------------------------------------------------
# Keyword Templates
# ---------------------------------------------------------------------------

KEYWORDS = ["救命", "帮助", "报警"]
KEYWORD_PINYIN = {
    "救命": "jiu_ming",
    "帮助": "bang_zhu",
    "报警": "bao_jing",
}

# Formant frequencies for each keyword's syllables (Hz)
# These approximate the F1/F2 patterns of Chinese syllables
KEYWORD_FORMANTS = {
    "救命": [
        # "jiu" — high front vowel, rising tone
        {"f0_start": 220, "f0_end": 280, "f1": 300, "f2": 2200, "duration": 0.35},
        # "ming" — nasal, falling tone
        {"f0_start": 280, "f0_end": 180, "f1": 400, "f2": 1800, "duration": 0.40},
    ],
    "帮助": [
        # "bang" — back vowel, level tone
        {"f0_start": 200, "f0_end": 210, "f1": 700, "f2": 1200, "duration": 0.35},
        # "zhu" — high back vowel, falling tone
        {"f0_start": 210, "f0_end": 150, "f1": 300, "f2": 800, "duration": 0.30},
    ],
    "报警": [
        # "bao" — diphthong, falling tone
        {"f0_start": 250, "f0_end": 180, "f1": 600, "f2": 1000, "duration": 0.30},
        # "jing" — high front, rising tone
        {"f0_start": 180, "f0_end": 250, "f1": 350, "f2": 2300, "duration": 0.35},
    ],
}


def synth_keyword_pcm(keyword: str, rng: np.random.Generator,
                      sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize a keyword utterance using formant synthesis.

    Creates a simplified vowel-consonant pattern that mimics the
    spectral characteristics of the target Chinese keyword.
    """
    formants = KEYWORD_FORMANTS[keyword]
    audio = []

    for syl in formants:
        num_samples = int(syl["duration"] * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # F0 pitch contour (linear interpolation)
        f0 = np.linspace(syl["f0_start"], syl["f0_end"], num_samples)

        # Generate glottal source (sum of harmonics)
        source = np.zeros(num_samples, dtype=np.float32)
        for harmonic in range(1, 6):
            freq = f0 * harmonic
            phase = np.cumsum(2 * np.pi * freq / sample_rate)
            amplitude = 1.0 / harmonic
            source += amplitude * np.sin(phase)

        # Apply formant filtering (simplified: bandpass around F1 and F2)
        # Use resonant filter approximation
        f1 = syl["f1"]
        f2 = syl["f2"]

        # Create formant envelope (two resonant peaks)
        freqs = np.fft.rfftfreq(num_samples, 1.0 / sample_rate)
        envelope = np.zeros_like(freqs)
        envelope += np.exp(-0.5 * ((freqs - f1) / 150) ** 2)  # F1 peak
        envelope += 0.7 * np.exp(-0.5 * ((freqs - f2) / 200) ** 2)  # F2 peak

        # Apply envelope in frequency domain
        spectrum = np.fft.rfft(source)
        filtered = spectrum * envelope
        syllable = np.fft.irfft(filtered, n=num_samples)

        # Apply amplitude envelope (attack-sustain-release)
        env = np.ones(num_samples, dtype=np.float32)
        attack = int(0.05 * num_samples)
        release = int(0.15 * num_samples)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            env[-release:] = np.linspace(1, 0, release)

        audio.append(syllable * env)

    # Concatenate syllables with short gap
    gap = np.zeros(int(0.03 * sample_rate), dtype=np.float32)
    full_audio = np.concatenate([audio[0], gap, audio[1]])

    # Add subtle noise (microphone simulation)
    noise = rng.normal(0, 0.005, len(full_audio)).astype(np.float32)
    full_audio = full_audio + noise

    # Normalize
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = full_audio / peak * 0.7

    return (full_audio * 32767).astype(np.int16)


def generate_keyword_templates(output_dir: str = "data/keywords",
                               samples_per_keyword: int = 5):
    """Generate synthetic keyword templates and save as WAV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    all_mfcc = {}

    for kw in KEYWORDS:
        templates = []
        for i in range(samples_per_keyword):
            # Vary synthesis slightly for each sample
            rng_i = np.random.default_rng(42 + hash(kw) % 1000 + i)
            pcm = synth_keyword_pcm(kw, rng_i)
            wav_path = output_path / f"template_{kw}_{i}.wav"
            write_wav_mono(str(wav_path), pcm)
            mfcc = extract_mfcc(pcm)
            templates.append(mfcc)
        all_mfcc[kw] = templates

    # Save MFCC templates as npz
    template_path = output_path / "keyword_templates.npz"
    save_data = {}
    for kw, templates in all_mfcc.items():
        for i, mfcc in enumerate(templates):
            save_data[f"{kw}_{i}"] = mfcc
    np.savez(str(template_path), **save_data)
    print(f"Generated {samples_per_keyword} templates × {len(KEYWORDS)} keywords")
    print(f"  Templates saved to: {template_path}")
    print(f"  WAV files saved to: {output_path}/")


def load_templates(template_path: str = "data/keywords/keyword_templates.npz"
                   ) -> dict[str, list[np.ndarray]]:
    """Load keyword templates from npz file."""
    data = np.load(template_path, allow_pickle=True)
    templates = {}
    for kw in KEYWORDS:
        templates[kw] = []
        for key in data.files:
            if key.startswith(kw + "_"):
                templates[kw].append(data[key])
    return templates


# ---------------------------------------------------------------------------
# Keyword Detection
# ---------------------------------------------------------------------------

def detect_keyword(pcm: np.ndarray,
                   templates: dict[str, list[np.ndarray]],
                   threshold: float = 50.0,
                   min_energy: float = 0.01) -> tuple[str | None, float, dict]:
    """Detect keyword from PCM audio using MFCC + DTW.

    Args:
        pcm: int16 mono audio samples
        templates: keyword templates dict
        threshold: max DTW distance to accept a match
        min_energy: minimum RMS energy to consider as speech (0-1 scale)

    Returns:
        (keyword, confidence, all_distances)
        keyword: matched keyword name, or None if no match
        confidence: 1.0 - normalized_distance (0.0 to 1.0)
        all_distances: dict of {keyword: min_dtw_distance}
    """
    # Energy-based pre-filter: reject silence/low-energy noise
    x = pcm.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < min_energy:
        # Too quiet to be speech — return all-inf distances
        all_distances = {kw: float("inf") for kw in templates}
        return None, 0.0, all_distances

    mfcc = extract_mfcc(pcm)

    all_distances = {}
    for kw, kw_templates in templates.items():
        # Match against all templates, take minimum distance
        min_dist = float("inf")
        for tmpl in kw_templates:
            dist = dtw_distance(mfcc, tmpl)
            if dist < min_dist:
                min_dist = dist
        all_distances[kw] = min_dist

    # Find best match
    best_kw = min(all_distances, key=all_distances.get)
    best_dist = all_distances[best_kw]

    if best_dist < threshold:
        # Convert distance to confidence (linear decay)
        confidence = max(0.0, 1.0 - best_dist / threshold)
        return best_kw, confidence, all_distances
    else:
        return None, 0.0, all_distances


# ---------------------------------------------------------------------------
# C Code Export
# ---------------------------------------------------------------------------

def export_templates_to_c(templates: dict[str, list[np.ndarray]],
                          output_path: str):
    """Export keyword MFCC templates as a C header file for embedded deployment."""
    lines = [
        "/* Auto-generated keyword templates for embedded C deployment.",
         " * MFCC features extracted at 16kHz, 13 coefficients per frame.",
         " * Each keyword has multiple template samples for DTW matching.",
         " */",
        "#ifndef KEYWORD_TEMPLATES_H",
        "#define KEYWORD_TEMPLATES_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define KW_NUM_MFCC     {NUM_MFCC}",
        f"#define KW_NUM_KEYWORDS  {len(KEYWORDS)}",
        "",
    ]

    # Keyword name table
    lines.append("/* Keyword names */")
    lines.append(f"static const char *const KW_NAMES[KW_NUM_KEYWORDS] = {{")
    for kw in KEYWORDS:
        lines.append(f'    "{kw}",')
    lines.append("};")
    lines.append("")

    # Export each template as a flat array
    total_bytes = 0
    for kw_idx, kw in enumerate(KEYWORDS):
        kw_templates = templates[kw]
        lines.append(f"/* {kw} — {len(kw_templates)} templates */")

        for t_idx, tmpl in enumerate(kw_templates):
            flat = tmpl.flatten().astype(np.float32)
            total_bytes += len(flat) * 4

            lines.append(f"static const float KW_TMPL_{kw_idx}_{t_idx}[] = {{")
            # 4 values per line
            for i in range(0, len(flat), 4):
                vals = flat[i:i + 4]
                lines.append("    " + ", ".join(f"{v:.6f}f" for v in vals) + ",")
            lines.append("};")
            lines.append("")

        # Template pointer table for this keyword
        lines.append(f"static const float *const KW_TMPL_{kw_idx}[{len(kw_templates)}] = {{")
        for t_idx in range(len(kw_templates)):
            lines.append(f"    KW_TMPL_{kw_idx}_{t_idx},")
        lines.append("};")
        lines.append("")

        # Frame counts
        frame_counts = [str(t.shape[0]) for t in kw_templates]
        lines.append(f"static const int KW_TMPL_{kw_idx}_FRAMES[{len(kw_templates)}] = {{{', '.join(frame_counts)}}};")
        lines.append("")

    # Master table
    lines.append("/* Master template table */")
    lines.append("static const float *const *KW_ALL_TMPL[KW_NUM_KEYWORDS] = {")
    for kw_idx in range(len(KEYWORDS)):
        lines.append(f"    KW_TMPL_{kw_idx},")
    lines.append("};")
    lines.append("")
    lines.append("static const int *KW_ALL_FRAMES[KW_NUM_KEYWORDS] = {")
    for kw_idx in range(len(KEYWORDS)):
        lines.append(f"    KW_TMPL_{kw_idx}_FRAMES,")
    lines.append("};")
    lines.append("")

    # Template counts
    counts = [str(len(templates[kw])) for kw in KEYWORDS]
    lines.append(f"static const int KW_TMPL_COUNTS[KW_NUM_KEYWORDS] = {{{', '.join(counts)}}};")
    lines.append("")

    lines.append(f"/* Total template memory: {total_bytes} bytes ({total_bytes / 1024:.1f} KB) */")
    lines.append("")
    lines.append("#endif /* KEYWORD_TEMPLATES_H */")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"C template header saved to: {output_path}")
    print(f"  Total template memory: {total_bytes} bytes ({total_bytes / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(templates: dict[str, list[np.ndarray]],
                   output_dir: str = "results"):
    """Run full keyword detection evaluation."""
    rng = np.random.default_rng(99)
    sr = SAMPLE_RATE

    # Generate test samples (different from training templates)
    test_samples = []
    for kw in KEYWORDS:
        for i in range(10):
            rng_i = np.random.default_rng(99 + hash(kw) % 1000 + i * 7)
            pcm = synth_keyword_pcm(kw, rng_i)
            test_samples.append((kw, pcm))

    # Add 10 background/noise samples (should not match any keyword)
    for i in range(10):
        rng_i = np.random.default_rng(99 + 700 + i)
        pcm = rng_i.normal(0, 100, sr).astype(np.int16)
        test_samples.append(("none", pcm))

    # Test with different thresholds
    results = {"per_keyword": {}, "thresholds": []}

    for threshold in [30, 40, 50, 60, 70]:
        correct = 0
        total = len(test_samples)
        per_kw = {kw: {"correct": 0, "total": 0} for kw in KEYWORDS + ["none"]}
        confusion = {}

        for true_kw, pcm in test_samples:
            detected, conf, all_dist = detect_keyword(pcm, templates, threshold)
            detected_str = detected if detected else "none"

            if detected_str == true_kw:
                correct += 1
                per_kw[true_kw]["correct"] += 1
            per_kw[true_kw]["total"] += 1

            key = f"{true_kw}->{detected_str}"
            confusion[key] = confusion.get(key, 0) + 1

        accuracy = correct / total
        results["thresholds"].append({
            "threshold": threshold,
            "accuracy": accuracy,
            "per_keyword": {k: v["correct"] / v["total"] for k, v in per_kw.items()},
            "confusion": confusion,
        })

    # Best threshold analysis
    best = max(results["thresholds"], key=lambda x: x["accuracy"])
    results["best_threshold"] = best["threshold"]
    results["best_accuracy"] = best["accuracy"]

    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    report_path = Path(output_dir) / "keyword_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  Keyword Detection Evaluation")
    print("=" * 60)
    print(f"  Keywords: {', '.join(KEYWORDS)}")
    print(f"  Templates per keyword: {len(templates[KEYWORDS[0]])}")
    print(f"  Test samples: {len(test_samples)} ({len(KEYWORDS)*10} keyword + 10 noise)")
    print()

    for r in results["thresholds"]:
        print(f"  Threshold={r['threshold']:3d}  Accuracy={r['accuracy']:.1%}  ", end="")
        per_kw_str = "  ".join(f"{k}:{v:.0%}" for k, v in r["per_keyword"].items())
        print(per_kw_str)

    print(f"\n  Best: threshold={best['threshold']}, accuracy={best['accuracy']:.1%}")
    print(f"\n  Confusion matrix (best threshold):")
    for key, count in sorted(best["confusion"].items()):
        parts = key.split("->")
        marker = " [OK]" if len(parts) == 2 and parts[0].strip() == parts[1].strip() else " [MISS]"
        print(f"    {key:25s}: {count:2d}{marker}")

    print(f"\n  Report saved to: {report_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Keyword Detection — MFCC + DTW")
    parser.add_argument("--generate_templates", action="store_true",
                        help="Generate synthetic keyword templates")
    parser.add_argument("--detect", type=str, help="Detect keyword from WAV file")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run full evaluation")
    parser.add_argument("--export_c", type=str,
                        help="Export templates to C header file")
    parser.add_argument("--samples_per_keyword", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--template_dir", default="data/keywords")
    args = parser.parse_args()

    if args.generate_templates:
        generate_keyword_templates(args.template_dir, args.samples_per_keyword)

    if args.detect or args.evaluate or args.export_c:
        template_path = Path(args.template_dir) / "keyword_templates.npz"
        if not template_path.exists():
            print(f"Templates not found at {template_path}, generating...")
            generate_keyword_templates(args.template_dir, args.samples_per_keyword)
        templates = load_templates(str(template_path))

    if args.detect:
        pcm = read_wav_mono(args.detect)
        # Trim or pad to ~1 second
        if len(pcm) > SAMPLE_RATE:
            pcm = pcm[:SAMPLE_RATE]
        elif len(pcm) < SAMPLE_RATE // 2:
            print(f"Warning: audio too short ({len(pcm)} samples)")

        keyword, confidence, all_dist = detect_keyword(pcm, templates, args.threshold)
        print(f"\nInput: {args.detect}")
        print(f"Detected: {keyword or 'none'}")
        print(f"Confidence: {confidence:.1%}")
        print(f"DTW distances:")
        for kw, dist in sorted(all_dist.items(), key=lambda x: x[1]):
            marker = " ← BEST" if kw == keyword else ""
            print(f"  {kw}: {dist:.2f}{marker}")

    if args.evaluate:
        run_evaluation(templates)

    if args.export_c:
        export_templates_to_c(templates, args.export_c)


if __name__ == "__main__":
    main()
