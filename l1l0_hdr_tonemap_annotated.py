#!/usr/bin/env python3
"""
带注释版 HDR-target L1-L0 tone mapping 脚本。

这个脚本和 l1l0_hdr_tonemap.py 的目标一致：
1. 复用原 CVPR 2018 L1-L0 分层分解，把 HDR 图像压缩成一个 tone-control 图。
2. 不输出 0..1 的 8-bit SDR，而是把结果映射到 0..peak 的浮点 HDR 范围。
3. 普通主体亮度落在 diffuse_white 附近，只让高光进入 diffuse_white..peak 的 HDR 头部空间。

默认输出仍然是 Radiance .hdr 文件。注意 .hdr/RGBE 没有 HDR10/PQ 元数据，
它只是保存浮点动态范围；具体显示效果取决于查看器和系统 HDR 映射。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# OpenCV 默认可能关闭 OpenEXR 支持；需要在 import cv2 前设置这个环境变量。
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

# 这里复用原 SDR 复现脚本里的基础算法函数，避免复制 L1-L0/TV 分解的大段代码。
# 本文件主要改的是“最后如何把分解结果映射到目标 HDR 亮度范围”。
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


# Rec.709 / sRGB 常用亮度权重。
# 用它从 RGB 估计亮度 Y，再按 Y 的比例重建 RGB，可以比 HSV 的 V 通道更自然。
BT709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def read_rgb_float(path: Path) -> np.ndarray:
    """读取 HDR/EXR/普通图像，并统一转成 RGB float64。"""
    # OpenCV 读彩色图默认是 BGR，不是 RGB。
    bgr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # 如果输入是 8-bit/16-bit 普通图，先归一化到 0..1。
    # 如果输入本来就是 float HDR/EXR，就保留原始动态范围。
    original_dtype = bgr.dtype
    bgr = bgr.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        bgr /= np.iinfo(original_dtype).max

    # 对 3 通道图，BGR -> RGB 只需要反转最后一个维度。
    # 这里不用 cv2.cvtColor，是因为 OpenCV 的 cvtColor 对 float64 支持有限。
    return bgr[..., ::-1]


def write_rgb_hdr(path: Path, rgb: np.ndarray) -> None:
    """把 RGB float 图写成 .hdr/.exr 等 OpenCV 支持的 HDR 格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCV 写彩色图需要 BGR 顺序，并且 HDR/EXR 写入通常使用 float32。
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
    """
    用原论文的分层方式生成 0..1 tone-control 图。

    tone-control 不是最终亮度，而是一个排序/结构信号：
    - 暗部接近 0
    - 主体和中间调在中间
    - 高光接近 1

    后面的 HDR 曲线会根据这个信号分配 0..peak 的输出亮度。
    """
    if lambda2 is None:
        lambda2 = lambda1 * 0.01

    # 原 MATLAB demo 用 HSV 的 V 通道做分解输入：
    # hdr_l = log(V + 0.0001)，再 normalize 到 0..1。
    hsv = rgb_to_hsv(hdr_rgb)
    value = normalize01(np.log(hsv[..., 2] + 0.0001))

    # L1-L0 两层分解：
    # d1: 细节层
    # d2: 中尺度细节层
    # b2: 基础层
    d1, d2, b2 = layer_decompose(value, lambda1, lambda2, lambda3)

    # 细节增强。sigma_d1 是细节层最大值，和原 demo 的 R_func(D1, 0, sigma_D1, 0.8, 1) 对齐。
    sigma_d1 = float(np.max(d1))
    if sigma_d1 <= EPS:
        d1_scaled = np.zeros_like(d1)
    else:
        d1_scaled = response_func(d1, 0.0, sigma_d1, 0.8, 1.0)

    # 基础层压缩，和原 demo 的 compress(B2, 2.2, 1) 对齐。
    # 这里把极少数负值夹到 0，避免 gamma/power 出现无效值。
    b2_compressed = compress_layer(np.maximum(b2, 0.0), 2.2, 1.0)

    # 原 demo 的合成方式：
    # hdr_lnn = 0.8*B2_n + D2 + 1.2*D1s
    tone_control = 0.8 * b2_compressed + d2 + 1.2 * d1_scaled

    # 去掉极端离群值，再归一化成 0..1。
    # 注意这一步只是生成 tone-control，不代表最终输出是 SDR。
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
    """
    把 0..1 tone-control 映射到 0..peak 的目标 HDR 亮度。

    参数直觉：
    - peak: 输出最大值，例如 10。
    - diffuse_white: 普通白点，建议 0.8..1.2；不要直接让普通白到 10。
    - knee: 从哪里开始进入 HDR 高光区。默认 0.75，表示只有最亮 25% 控制量进高光头部。
    - midtone_gamma: 中间调曲线，越大主体越暗。
    - highlight_gamma: 高光 rolloff，越大越晚接近 peak。
    - black_lift: 黑位抬升，默认 0。
    """
    if not 0.0 < knee < 1.0:
        raise ValueError("--knee must be between 0 and 1")
    if not 0.0 < diffuse_white <= peak:
        raise ValueError("--diffuse-white must be > 0 and <= --peak")
    if black_lift < 0.0:
        raise ValueError("--black-lift must be >= 0")

    x = np.clip(x, 0.0, 1.0)
    y = np.empty_like(x, dtype=np.float64)

    # 低/中亮度区域：0..knee 映射到 black_lift..diffuse_white。
    # 这保证了大部分画面不会因为 peak=10 而整体变亮。
    low = x <= knee
    low_x = x[low] / knee
    y[low] = black_lift + (diffuse_white - black_lift) * np.power(low_x, midtone_gamma)

    # 高光区域：knee..1 映射到 diffuse_white..peak。
    # 只有 tone-control 足够高的区域才使用 HDR 头部空间。
    high_t = (x[~low] - knee) / (1.0 - knee)
    y[~low] = diffuse_white + (peak - diffuse_white) * np.power(high_t, highlight_gamma)

    return np.clip(y, 0.0, peak)


def reconstruct_rgb_from_luminance(
    source_rgb: np.ndarray,
    target_y: np.ndarray,
    peak: float,
    saturation: float,
) -> np.ndarray:
    """
    根据目标亮度 target_y 重建 RGB。

    原 SDR demo 用 HSV(H, S*0.6, newV) 重建，会改变颜色模型。
    这里改成亮度比例法：
        out_rgb = source_rgb / source_y * target_y

    这样可以较好保持原始色相。saturation 用来控制颜色强度，避免 HDR 下颜色过饱和。
    """
    if saturation < 0.0:
        raise ValueError("--saturation must be >= 0")

    # HDR/EXR 里有时会有小负值；显示图像时没有物理意义，先夹到非负。
    source_rgb = np.maximum(source_rgb, 0.0)

    # 用 Rec.709 权重估计输入亮度。
    source_y = np.tensordot(source_rgb, BT709_LUMA, axes=([-1], [0]))

    # 彩色重建：保持 RGB 相对比例，只把亮度改成 target_y。
    colored = source_rgb * (target_y[..., None] / np.maximum(source_y[..., None], EPS))

    # 灰度版本用于降低饱和度：saturation=1 保持彩色，saturation=0 变成灰度亮度图。
    gray = target_y[..., None]
    out = gray + saturation * (colored - gray)

    # 清理 NaN/Inf，并限制到非负。
    out = np.nan_to_num(out, nan=0.0, posinf=peak, neginf=0.0)
    out = np.maximum(out, 0.0)

    # 如果某些高饱和颜色导致单个通道超过 peak，就按像素整体缩放，避免硬裁剪改色相。
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
    """
    HDR tone mapping 主流程。

    数据流：
    HDR RGB -> L1-L0 tone-control -> HDR 目标亮度曲线 -> 亮度比例法重建 RGB。
    """
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
    """打印输出图的关键统计，方便判断是不是整体太亮。"""
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

    # HDR 输出曲线参数。
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

    # 分位裁剪和 L1-L0 分解参数。一般先别动 lambda，优先调 HDR 曲线参数。
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
