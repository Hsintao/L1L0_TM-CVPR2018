#!/usr/bin/env python3
"""
Single-file Python reproduction of:

    Z. Liang, J. Xu, D. Zhang, Z. Cao, L. Zhang,
    "A Hybrid L1-L0 Layer Decomposition Model for Tone Mapping", CVPR 2018.

The original MATLAB demo reads an HDR image, decomposes the log luminance with
hybrid L1-L0 and TV-like layers, rescales the detail/base layers, then writes an
LDR tone-mapped RGB image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage


EPS = 1e-12


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "python":
        return script_dir.parent
    return script_dir


def resolve_input_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    rooted = root / path
    if rooted.exists():
        return rooted
    return path


def matlab_round(x: np.ndarray | float) -> np.ndarray | int:
    """MATLAB round for non-negative values: halves round away from zero."""
    out = np.floor(np.asarray(x) + 0.5).astype(np.int64)
    if out.ndim == 0:
        return int(out)
    return out


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    denom = xmax - xmin
    if abs(denom) < EPS:
        return np.zeros_like(x, dtype=np.float64)
    return (x - xmin) / denom


def clamp_percentile_like_matlab(x: np.ndarray, low_frac: float, high_frac: float) -> np.ndarray:
    """Match clampp.m, including MATLAB's sorted-index percentile behavior."""
    flat = np.sort(np.asarray(x, dtype=np.float64).ravel())
    total = flat.size
    low_index = max(1, min(total, matlab_round(low_frac * total))) - 1
    high_index = max(1, min(total, matlab_round(high_frac * total))) - 1
    low = flat[low_index]
    high = flat[high_index]
    return np.clip(x, low, high)


def compress_layer(x: np.ndarray, gamma: float, width: float = 1.0) -> np.ndarray:
    return width * np.power(x / width, 1.0 / gamma)


def response_func(x: np.ndarray, g: float, sigma: float, a: float, b: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    index_low = np.abs(x - g) <= sigma
    temp = x[index_low]
    y[index_low] = g + np.power(np.abs(temp - g) / sigma, a) * sigma * np.sign(temp - g)

    index_high = ~index_low
    temp = x[index_high]
    y[index_high] = g + (b * (np.abs(temp - g) - sigma) + sigma) * np.sign(temp - g)
    return y


def psf2otf(psf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """MATLAB psf2otf equivalent for 2-D kernels."""
    psf = np.asarray(psf, dtype=np.float64)
    padded = np.zeros(out_shape, dtype=np.float64)
    slices = tuple(slice(0, size) for size in psf.shape)
    padded[slices] = psf
    for axis, axis_size in enumerate(psf.shape):
        padded = np.roll(padded, -int(axis_size // 2), axis=axis)
    return np.fft.fft2(padded)


def grad_forward_x(x: np.ndarray) -> np.ndarray:
    return np.roll(x, -1, axis=1) - x


def grad_forward_y(x: np.ndarray) -> np.ndarray:
    return np.roll(x, -1, axis=0) - x


def shrink(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def l1_l0_decompose(s: np.ndarray, lambda1: float, lambda2: float, iterations: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid L1-L0 decomposition. Returns detail D and base B."""
    s = np.asarray(s, dtype=np.float64)
    height, width = s.shape

    otf_fx = psf2otf(np.array([[1.0, -1.0]]), (height, width))
    otf_fy = psf2otf(np.array([[1.0], [-1.0]]), (height, width))
    dxdy = np.abs(otf_fy) ** 2 + np.abs(otf_fx) ** 2
    fft_s = np.fft.fft2(s)

    b = s.copy()
    cx = np.zeros_like(s)
    cy = np.zeros_like(s)
    ex = np.zeros_like(s)
    ey = np.zeros_like(s)
    l1x = np.zeros_like(s)
    l1y = np.zeros_like(s)
    l2x = np.zeros_like(s)
    l2y = np.zeros_like(s)
    rho1 = 1.0
    rho2 = 1.0

    diff_sx = grad_forward_x(s)
    diff_sy = grad_forward_y(s)

    for _ in range(iterations):
        clx = cx + l1x / rho1
        cly = cy + l1y / rho1
        elx = diff_sx - ex - l2x / rho2
        ely = diff_sy - ey - l2y / rho2

        nominator = (
            fft_s
            + rho1 * np.conj(otf_fx) * np.fft.fft2(clx)
            + rho1 * np.conj(otf_fy) * np.fft.fft2(cly)
            + rho2 * np.conj(otf_fx) * np.fft.fft2(elx)
            + rho2 * np.conj(otf_fy) * np.fft.fft2(ely)
        )
        denominator = 1.0 + (rho1 + rho2) * dxdy
        b_new = np.real(np.fft.ifft2(nominator / denominator))

        diff_bx = grad_forward_x(b_new)
        diff_by = grad_forward_y(b_new)

        cx_new = shrink(diff_bx - l1x / rho1, lambda1 / rho1)
        cy_new = shrink(diff_by - l1y / rho1, lambda1 / rho1)

        bx = diff_sx - diff_bx - l2x / rho2
        by = diff_sy - diff_by - l2y / rho2
        ex_new = bx.copy()
        ey_new = by.copy()
        ex_new[bx * bx < 2.0 * lambda2 / rho2] = 0.0
        ey_new[by * by < 2.0 * lambda2 / rho2] = 0.0

        l1x_new = l1x + rho1 * (cx_new - diff_bx)
        l1y_new = l1y + rho1 * (cy_new - diff_by)
        l2x_new = l2x + rho2 * (ex_new - diff_sx + diff_bx)
        l2y_new = l2y + rho2 * (ey_new - diff_sy + diff_by)

        b = b_new
        cx, cy = cx_new, cy_new
        ex, ey = ex_new, ey_new
        l1x, l1y = l1x_new, l1y_new
        l2x, l2y = l2x_new, l2y_new
        rho1 *= 4.0
        rho2 *= 4.0

    return s - b, b


def l1_decompose(s: np.ndarray, lambda_: float, iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """TV-like L1 decomposition. Returns detail D and base B."""
    s = np.asarray(s, dtype=np.float64)
    height, width = s.shape

    otf_fx = psf2otf(np.array([[1.0, -1.0]]), (height, width))
    otf_fy = psf2otf(np.array([[1.0], [-1.0]]), (height, width))
    dxdy = np.abs(otf_fy) ** 2 + np.abs(otf_fx) ** 2
    fft_s = np.fft.fft2(s)

    b = s.copy()
    c = np.zeros_like(s)
    d = np.zeros_like(s)
    t1 = np.zeros_like(s)
    t2 = np.zeros_like(s)
    rho = 1.0

    for _ in range(iterations):
        nominator = (
            fft_s
            + rho * np.conj(otf_fx) * np.fft.fft2(c + t1 / rho)
            + rho * np.conj(otf_fy) * np.fft.fft2(d + t2 / rho)
        )
        denominator = 1.0 + rho * dxdy
        b_new = np.real(np.fft.ifft2(nominator / denominator))

        grad_x = grad_forward_x(b_new)
        grad_y = grad_forward_y(b_new)

        c_new = shrink(grad_x - t1 / rho, lambda_ / rho)
        d_new = shrink(grad_y - t2 / rho, lambda_ / rho)
        t1_new = t1 + rho * (c_new - grad_x)
        t2_new = t2 + rho * (d_new - grad_y)

        b = b_new
        c, d = c_new, d_new
        t1, t2 = t1_new, t2_new
        rho *= 2.0

    return s - b, b


def bilateral_filter_grid(
    data: np.ndarray,
    edge: np.ndarray | None = None,
    edge_min: float | None = None,
    edge_max: float | None = None,
    sigma_spatial: float | None = None,
    sigma_range: float | None = None,
    sampling_spatial: float | None = None,
    sampling_range: float | None = None,
) -> np.ndarray:
    """Bilateral-grid implementation matching bilateralFilter.m."""
    data = np.asarray(data, dtype=np.float64)
    if edge is None:
        edge = data
    edge = np.asarray(edge, dtype=np.float64)
    if data.shape != edge.shape:
        raise ValueError("data and edge must have the same shape")

    input_height, input_width = data.shape
    if edge_min is None:
        edge_min = float(np.nanmin(edge))
    if edge_max is None:
        edge_max = float(np.nanmax(edge))
    edge_delta = edge_max - edge_min

    if sigma_spatial is None:
        sigma_spatial = min(input_width, input_height) / 16.0
    if sigma_range is None:
        sigma_range = 0.1 * edge_delta
    if sampling_spatial is None:
        sampling_spatial = sigma_spatial
    if sampling_range is None:
        sampling_range = sigma_range

    derived_sigma_spatial = sigma_spatial / sampling_spatial
    derived_sigma_range = sigma_range / sampling_range
    padding_xy = int(np.floor(2.0 * derived_sigma_spatial) + 1)
    padding_z = int(np.floor(2.0 * derived_sigma_range) + 1)

    downsampled_width = int(np.floor((input_width - 1) / sampling_spatial) + 1 + 2 * padding_xy)
    downsampled_height = int(np.floor((input_height - 1) / sampling_spatial) + 1 + 2 * padding_xy)
    downsampled_depth = int(np.floor(edge_delta / sampling_range) + 1 + 2 * padding_z)

    grid_data = np.zeros((downsampled_height, downsampled_width, downsampled_depth), dtype=np.float64)
    grid_weights = np.zeros_like(grid_data)

    ii, jj = np.meshgrid(np.arange(input_height), np.arange(input_width), indexing="ij")
    di = matlab_round(ii / sampling_spatial) + padding_xy
    dj = matlab_round(jj / sampling_spatial) + padding_xy
    dz = matlab_round((edge - edge_min) / sampling_range) + padding_z
    dz = np.clip(dz, 0, downsampled_depth - 1)

    valid = ~np.isnan(data)
    np.add.at(grid_data, (di[valid], dj[valid], dz[valid]), data[valid])
    np.add.at(grid_weights, (di[valid], dj[valid], dz[valid]), 1.0)

    kernel_width = int(2.0 * derived_sigma_spatial + 1.0)
    kernel_height = kernel_width
    kernel_depth = int(2.0 * derived_sigma_range + 1.0)
    half_width = int(np.floor(kernel_width / 2.0))
    half_height = int(np.floor(kernel_height / 2.0))
    half_depth = int(np.floor(kernel_depth / 2.0))

    gy, gx, gz = np.meshgrid(
        np.arange(kernel_height),
        np.arange(kernel_width),
        np.arange(kernel_depth),
        indexing="ij",
    )
    gx = gx - half_width
    gy = gy - half_height
    gz = gz - half_depth
    grid_r_squared = (
        (gx * gx + gy * gy) / (derived_sigma_spatial * derived_sigma_spatial)
        + (gz * gz) / (derived_sigma_range * derived_sigma_range)
    )
    kernel = np.exp(-0.5 * grid_r_squared)

    blurred_grid_data = ndimage.convolve(grid_data, kernel, mode="constant", cval=0.0)
    blurred_grid_weights = ndimage.convolve(grid_weights, kernel, mode="constant", cval=0.0)

    normalized_blurred_grid = np.zeros_like(blurred_grid_data)
    np.divide(
        blurred_grid_data,
        blurred_grid_weights,
        out=normalized_blurred_grid,
        where=blurred_grid_weights != 0,
    )

    coords = np.vstack(
        [
            (ii.ravel() / sampling_spatial) + padding_xy,
            (jj.ravel() / sampling_spatial) + padding_xy,
            ((edge.ravel() - edge_min) / sampling_range) + padding_z,
        ]
    )
    out = ndimage.map_coordinates(
        normalized_blurred_grid,
        coords,
        order=1,
        mode="nearest",
    )
    return out.reshape(data.shape)


def layer_decompose(
    img: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    scale: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = img.shape
    d1, b1 = l1_l0_decompose(img, lambda1, lambda2)

    small_height = matlab_round(height / scale)
    small_width = matlab_round(width / scale)
    b1_down = cv2.resize(b1, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    _, b2_down = l1_decompose(b1_down, lambda3)
    b2_resized = cv2.resize(b2_down, (width, height), interpolation=cv2.INTER_LINEAR)
    b2 = bilateral_filter_grid(
        b2_resized,
        normalize01(b1),
        0.0,
        1.0,
        min(width, height) / 100.0,
        0.05,
    )
    d2 = b1 - b2
    return d1, d2, b2


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    nonzero = delta > EPS
    mask = nonzero & (maxc == r)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0
    mask = nonzero & (maxc == g)
    h[mask] = (b[mask] - r[mask]) / delta[mask] + 2.0
    mask = nonzero & (maxc == b)
    h[mask] = (r[mask] - g[mask]) / delta[mask] + 4.0
    h /= 6.0

    s = np.zeros_like(maxc)
    positive = maxc > EPS
    s[positive] = delta[positive] / maxc[positive]
    v = maxc
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = np.asarray(hsv[..., 0], dtype=np.float64) % 1.0
    s = np.asarray(hsv[..., 1], dtype=np.float64)
    v = np.asarray(hsv[..., 2], dtype=np.float64)

    i = np.floor(h * 6.0).astype(np.int64)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    mod = i % 6
    out = np.empty(hsv.shape, dtype=np.float64)
    masks = [mod == k for k in range(6)]
    out[masks[0]] = np.stack([v[masks[0]], t[masks[0]], p[masks[0]]], axis=-1)
    out[masks[1]] = np.stack([q[masks[1]], v[masks[1]], p[masks[1]]], axis=-1)
    out[masks[2]] = np.stack([p[masks[2]], v[masks[2]], t[masks[2]]], axis=-1)
    out[masks[3]] = np.stack([p[masks[3]], q[masks[3]], v[masks[3]]], axis=-1)
    out[masks[4]] = np.stack([t[masks[4]], p[masks[4]], v[masks[4]]], axis=-1)
    out[masks[5]] = np.stack([v[masks[5]], p[masks[5]], q[masks[5]]], axis=-1)
    return out


def read_hdr_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)


def write_ldr_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = np.clip(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr8):
        raise OSError(f"Could not write image: {path}")


def tone_map(
    hdr_rgb: np.ndarray,
    lambda1: float = 0.3,
    lambda2: float | None = None,
    lambda3: float = 0.1,
) -> np.ndarray:
    if lambda2 is None:
        lambda2 = lambda1 * 0.01

    hsv = rgb_to_hsv(hdr_rgb)
    luminance = np.log(hsv[..., 2] + 0.0001)
    luminance = normalize01(luminance)

    d1, d2, b2 = layer_decompose(luminance, lambda1, lambda2, lambda3)

    sigma_d1 = float(np.max(d1))
    d1_scaled = response_func(d1, 0.0, sigma_d1, 0.8, 1.0)
    b2_compressed = compress_layer(b2, 2.2, 1.0)
    ldr_luminance = 0.8 * b2_compressed + d2 + 1.2 * d1_scaled

    ldr_luminance = normalize01(clamp_percentile_like_matlab(ldr_luminance, 0.005, 0.995))
    out_hsv = np.stack([hsv[..., 0], hsv[..., 1] * 0.6, ldr_luminance], axis=-1)
    return np.clip(hsv_to_rgb(out_hsv), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_input = root / "inputs" / "1.hdr"
    parser = argparse.ArgumentParser(
        description="Tone-map an HDR image with the CVPR 2018 hybrid L1-L0 decomposition demo."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(default_input),
        help="Input HDR image path. Default: inputs/1.hdr",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output LDR image path. Default: results/<input_stem>_l1l0.png",
    )
    parser.add_argument("--lambda1", type=float, default=0.3, help="Hybrid L1-L0 lambda1. Default: 0.3")
    parser.add_argument(
        "--lambda2",
        type=float,
        default=None,
        help="Hybrid L1-L0 lambda2. Default: lambda1 * 0.01",
    )
    parser.add_argument("--lambda3", type=float, default=0.1, help="Second-scale TV lambda. Default: 0.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    input_path = resolve_input_path(args.input, root)
    output_path = Path(args.output) if args.output else root / "results" / f"{input_path.stem}_l1l0.png"

    hdr_rgb = read_hdr_rgb(input_path)
    out_rgb = tone_map(hdr_rgb, args.lambda1, args.lambda2, args.lambda3)
    write_ldr_rgb(output_path, out_rgb)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
