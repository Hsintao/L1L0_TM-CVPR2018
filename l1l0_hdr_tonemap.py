#!/usr/bin/env python3
"""
HDR-target variant of the CVPR 2018 hybrid L1-L0 tone mapper.

Unlike l1l0_tonemap.py, this script does not make an 8-bit SDR image. It keeps
the L1-L0 layer decomposition, then maps the composed tone-control signal into a
floating-point display range such as 0..10. The lower/mid tones stay around the
diffuse-white region, while only highlights are rolled into the HDR headroom.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from l1l0_tonemap import (
    EPS,
    clamp_percentile_like_matlab,
    compress_layer,
    layer_decompose,
    normalize01,
    project_root,
    resolve_input_path,
    response_func,
    rgb_to_hsv,
)


BT709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def read_rgb_float(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    original_dtype = bgr.dtype
    bgr = bgr.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        bgr /= np.iinfo(original_dtype).max
    return bgr[..., ::-1]


def write_rgb_hdr(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Could not write image: {path}")


def build_tone_control(
    hdr_rgb: np.ndarray,
    lambda1: float,
    lambda2: float | None,
    lambda3: float,
    low_clip: float,
    high_clip: float,
) -> np.ndarray:
    """Run the original decomposition and produce a normalized tone-control map."""
    if lambda2 is None:
        lambda2 = lambda1 * 0.01

    hsv = rgb_to_hsv(hdr_rgb)
    value = normalize01(np.log(hsv[..., 2] + 0.0001))

    d1, d2, b2 = layer_decompose(value, lambda1, lambda2, lambda3)

    sigma_d1 = float(np.max(d1))
    if sigma_d1 <= EPS:
        d1_scaled = np.zeros_like(d1)
    else:
        d1_scaled = response_func(d1, 0.0, sigma_d1, 0.8, 1.0)

    b2_compressed = compress_layer(np.maximum(b2, 0.0), 2.2, 1.0)
    tone_control = 0.8 * b2_compressed + d2 + 1.2 * d1_scaled
    tone_control = clamp_percentile_like_matlab(tone_control, low_clip, high_clip)
    return normalize01(tone_control)


def hdr_luminance_curve(
    x: np.ndarray,
    peak: float,
    diffuse_white: float,
    knee: float,
    midtone_gamma: float,
    highlight_gamma: float,
    black_lift: float,
) -> np.ndarray:
    """Map normalized tone control to target HDR display luminance units."""
    if not 0.0 < knee < 1.0:
        raise ValueError("--knee must be between 0 and 1")
    if not 0.0 < diffuse_white <= peak:
        raise ValueError("--diffuse-white must be > 0 and <= --peak")
    if black_lift < 0.0:
        raise ValueError("--black-lift must be >= 0")

    x = np.clip(x, 0.0, 1.0)
    y = np.empty_like(x, dtype=np.float64)

    low = x <= knee
    low_x = x[low] / knee
    y[low] = black_lift + (diffuse_white - black_lift) * np.power(low_x, midtone_gamma)

    high_t = (x[~low] - knee) / (1.0 - knee)
    y[~low] = diffuse_white + (peak - diffuse_white) * np.power(high_t, highlight_gamma)
    return np.clip(y, 0.0, peak)


def reconstruct_rgb_from_luminance(
    source_rgb: np.ndarray,
    target_y: np.ndarray,
    peak: float,
    saturation: float,
) -> np.ndarray:
    """Preserve source chroma with a luminance-ratio reconstruction."""
    if saturation < 0.0:
        raise ValueError("--saturation must be >= 0")

    source_rgb = np.maximum(source_rgb, 0.0)
    source_y = np.tensordot(source_rgb, BT709_LUMA, axes=([-1], [0]))
    colored = source_rgb * (target_y[..., None] / np.maximum(source_y[..., None], EPS))
    gray = target_y[..., None]
    out = gray + saturation * (colored - gray)
    out = np.nan_to_num(out, nan=0.0, posinf=peak, neginf=0.0)
    out = np.maximum(out, 0.0)

    max_channel = np.max(out, axis=-1, keepdims=True)
    out *= np.minimum(1.0, peak / np.maximum(max_channel, EPS))
    return np.clip(out, 0.0, peak).astype(np.float32)


def tone_map_hdr(
    hdr_rgb: np.ndarray,
    lambda1: float = 0.3,
    lambda2: float | None = None,
    lambda3: float = 0.1,
    peak: float = 10.0,
    diffuse_white: float = 1.0,
    knee: float = 0.75,
    midtone_gamma: float = 1.6,
    highlight_gamma: float = 2.0,
    saturation: float = 0.85,
    black_lift: float = 0.0,
    low_clip: float = 0.005,
    high_clip: float = 0.995,
) -> np.ndarray:
    tone_control = build_tone_control(hdr_rgb, lambda1, lambda2, lambda3, low_clip, high_clip)
    target_y = hdr_luminance_curve(
        tone_control,
        peak=peak,
        diffuse_white=diffuse_white,
        knee=knee,
        midtone_gamma=midtone_gamma,
        highlight_gamma=highlight_gamma,
        black_lift=black_lift,
    )
    return reconstruct_rgb_from_luminance(hdr_rgb, target_y, peak=peak, saturation=saturation)


def image_stats(rgb: np.ndarray) -> str:
    return (
        f"shape={rgb.shape} dtype={rgb.dtype} "
        f"min={float(np.min(rgb)):.6g} max={float(np.max(rgb)):.6g} "
        f"mean={float(np.mean(rgb)):.6g} "
        f"p50={float(np.percentile(rgb, 50)):.6g} "
        f"p95={float(np.percentile(rgb, 95)):.6g} "
        f"p99={float(np.percentile(rgb, 99)):.6g}"
    )


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_input = root / "inputs" / "1.hdr"
    parser = argparse.ArgumentParser(
        description="Tone-map HDR input into a floating-point 0..peak HDR output."
    )
    parser.add_argument("input", nargs="?", default=str(default_input), help="Input HDR/EXR image path.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HDR image path. Default: results/<input_stem>_l1l0_hdr_peak<peak>.hdr",
    )
    parser.add_argument("--peak", type=float, default=10.0, help="Output peak value. Default: 10.0")
    parser.add_argument(
        "--diffuse-white",
        type=float,
        default=1.0,
        help="Where ordinary white lands before HDR highlight rolloff. Default: 1.0",
    )
    parser.add_argument(
        "--knee",
        type=float,
        default=0.75,
        help="Tone-control point where highlights start using HDR headroom. Default: 0.75",
    )
    parser.add_argument("--midtone-gamma", type=float, default=1.6, help="Lower/mid tone curve. Default: 1.6")
    parser.add_argument("--highlight-gamma", type=float, default=2.0, help="Highlight rolloff curve. Default: 2.0")
    parser.add_argument("--saturation", type=float, default=0.85, help="Chroma preservation strength. Default: 0.85")
    parser.add_argument("--black-lift", type=float, default=0.0, help="Minimum output luminance. Default: 0.0")
    parser.add_argument("--low-clip", type=float, default=0.005, help="Low percentile clamp. Default: 0.005")
    parser.add_argument("--high-clip", type=float, default=0.995, help="High percentile clamp. Default: 0.995")
    parser.add_argument("--lambda1", type=float, default=0.3, help="Hybrid L1-L0 lambda1. Default: 0.3")
    parser.add_argument("--lambda2", type=float, default=None, help="Hybrid L1-L0 lambda2. Default: lambda1 * 0.01")
    parser.add_argument("--lambda3", type=float, default=0.1, help="Second-scale TV lambda. Default: 0.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    input_path = resolve_input_path(args.input, root)
    output_path = (
        Path(args.output)
        if args.output
        else root / "results" / f"{input_path.stem}_l1l0_hdr_peak{args.peak:g}.hdr"
    )

    hdr_rgb = read_rgb_float(input_path)
    out_rgb = tone_map_hdr(
        hdr_rgb,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        peak=args.peak,
        diffuse_white=args.diffuse_white,
        knee=args.knee,
        midtone_gamma=args.midtone_gamma,
        highlight_gamma=args.highlight_gamma,
        saturation=args.saturation,
        black_lift=args.black_lift,
        low_clip=args.low_clip,
        high_clip=args.high_clip,
    )
    write_rgb_hdr(output_path, out_rgb)
    print(f"Wrote {output_path}")
    print(image_stats(out_rgb))


if __name__ == "__main__":
    main()
