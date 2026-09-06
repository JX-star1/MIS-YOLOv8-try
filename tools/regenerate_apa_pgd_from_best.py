
#!/usr/bin/env python3
"""Regenerate journal-grade APA and PGD visualizations from a trained best.pt.

This script is designed for the original PGAP-YOLO Ultralytics source tree. It
does not retrain or modify the network. Forward hooks capture:

1. PAFeature.weight logits and their concatenated aligned source features.
2. PGDHeatGate projected input, heat logits, and gated output.

The script exports native-resolution NumPy arrays, display-resolution arrays,
600-dpi PNG figures, vector-container PDF figures, and a JSON audit record.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import font_manager
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export APA weights and PGD responses from a trained PGAP-YOLO checkpoint."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to the final best.pt.")
    parser.add_argument("--image", type=Path, required=True, help="One validation image.")
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="Matching YOLO-format label file. Required for a new GT Gaussian heatmap.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--small-area",
        type=float,
        default=64.0**2,
        help="Maximum small-object area in letterboxed model-input pixels.",
    )
    parser.add_argument(
        "--eta-bins",
        type=float,
        nargs=3,
        default=(16.0, 32.0, 64.0),
        metavar=("B1", "B2", "B3"),
        help=(
            "Long-side thresholds in model-input pixels for eta={2,3,4,6}. "
            "Set these to exactly the values used by the PGD training target builder."
        ),
    )
    parser.add_argument(
        "--small-classes",
        type=int,
        nargs="*",
        default=(0, 1, 2, 6, 7, 9),
        help=(
            "Class IDs always included in the PGD target. The archived training code "
            "uses 0 1 2 6 7 9 for VisDrone."
        ),
    )
    parser.add_argument(
        "--map-style",
        choices=("raw", "overlay"),
        default="raw",
        help=(
            "Use raw maps for quantitatively faithful main-paper figures, or overlay "
            "maps on a grayscale image for qualitative supplementary figures."
        ),
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.58,
        help="Heatmap opacity when --map-style=overlay.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=600,
        help="Rasterization DPI used for PNG and raster layers embedded in PDF.",
    )
    parser.add_argument(
        "--pdf-min-ppi",
        type=int,
        default=300,
        help="Minimum accepted PPI for color/grayscale raster layers in PDF preflight.",
    )
    parser.add_argument(
        "--pgd-stride",
        type=int,
        default=None,
        help=(
            "Explicit PGD stride/level to visualize. Required when more than one active "
            "PGDHeatGate is captured, preventing silent selection of a favorable level."
        ),
    )
    return parser.parse_args()


def first_tensor(value: Any) -> torch.Tensor:
    """Return the first tensor in a nested hook input/output structure."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"No tensor found in object of type {type(value)!r}.")


def channel_rms(tensor: torch.Tensor) -> torch.Tensor:
    """Convert BxCxHxW features to a nonnegative BxHxW response map."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected BxCxHxW tensor, received shape {tuple(tensor.shape)}.")
    return tensor.float().square().mean(dim=1).sqrt()


def letterbox_geometry(
    original_hw: tuple[int, int], model_hw: tuple[int, int]
) -> tuple[float, int, int, int, int]:
    """Reconstruct the resize ratio and symmetric padding used by letterboxing."""
    original_h, original_w = original_hw
    model_h, model_w = model_hw
    ratio = min(model_h / original_h, model_w / original_w)
    resized_w = int(round(original_w * ratio))
    resized_h = int(round(original_h * ratio))
    left = int(round((model_w - resized_w) / 2.0 - 0.1))
    top = int(round((model_h - resized_h) / 2.0 - 0.1))
    return ratio, left, top, resized_w, resized_h


def map_to_original(
    feature_map: np.ndarray,
    model_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> np.ndarray:
    """Resize a feature-space map, remove letterbox padding, and restore image size."""
    if feature_map.ndim != 2:
        raise ValueError(f"Expected a 2-D map, received shape {feature_map.shape}.")
    model_h, model_w = model_hw
    original_h, original_w = original_hw
    _, left, top, resized_w, resized_h = letterbox_geometry(original_hw, model_hw)
    model_map = cv2.resize(feature_map, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    cropped = model_map[top : top + resized_h, left : left + resized_w]
    if cropped.size == 0:
        raise RuntimeError("Letterbox removal produced an empty map.")
    return cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)


def eta_from_long_side(long_side: float, bins: tuple[float, float, float]) -> float:
    """Map an object size to eta={2,3,4,6}; bins must match the training code."""
    if long_side < bins[0]:
        return 2.0
    if long_side < bins[1]:
        return 3.0
    if long_side < bins[2]:
        return 4.0
    return 6.0


def build_gaussian_target(
    label_path: Path,
    original_hw: tuple[int, int],
    model_hw: tuple[int, int],
    heat_hw: tuple[int, int],
    small_area: float,
    eta_bins: tuple[float, float, float],
    small_classes: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Build the category-agnostic PGD target from YOLO normalized boxes.

    The implementation follows the manuscript equation:
      sigma_x=max(w_feature/eta, 1), sigma_y=max(h_feature/eta, 1).

    IMPORTANT: eta_bins and the small-object rule must be identical to the
    training target builder. If the source code uses different intervals or
    class-specific selection, change this function before using the GT panel.
    """
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 5:
            continue
        rows.append([float(value) for value in fields[:5]])

    original_h, original_w = original_hw
    model_h, model_w = model_hw
    heat_h, heat_w = heat_hw
    ratio, left, top, _, _ = letterbox_geometry(original_hw, model_hw)
    stride_x = model_w / heat_w
    stride_y = model_h / heat_h
    grid_y, grid_x = np.mgrid[0:heat_h, 0:heat_w].astype(np.float32)
    target = np.zeros((heat_h, heat_w), dtype=np.float32)
    retained = 0

    for class_id, cx_n, cy_n, width_n, height_n in rows:
        cx_model = cx_n * original_w * ratio + left
        cy_model = cy_n * original_h * ratio + top
        width_model = width_n * original_w * ratio
        height_model = height_n * original_h * ratio
        if int(class_id) not in small_classes and width_model * height_model > small_area:
            continue
        if width_model <= 1.0 or height_model <= 1.0:
            continue

        retained += 1
        eta = eta_from_long_side(max(width_model, height_model), eta_bins)
        gx = cx_model / stride_x
        gy = cy_model / stride_y
        width_feature = width_model / stride_x
        height_feature = height_model / stride_y
        sigma_x = max(width_feature / eta, 1.0)
        sigma_y = max(height_feature / eta, 1.0)
        gaussian = np.exp(
            -0.5
            * (
                ((grid_x - gx) / sigma_x) ** 2
                + ((grid_y - gy) / sigma_y) ** 2
            )
        ).astype(np.float32)
        target = np.maximum(target, gaussian)

    return target, retained


def normalize_joint(
    first: np.ndarray, second: np.ndarray, lower: float = 1.0, upper: float = 99.0
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalize two response maps with one shared robust range."""
    values = np.concatenate([first.reshape(-1), second.reshape(-1)])
    low, high = np.percentile(values, [lower, upper])
    if not np.isfinite(high) or high <= low:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if high <= low:
        high = low + 1e-6
    first_n = np.clip((first - low) / (high - low), 0.0, 1.0)
    second_n = np.clip((second - low) / (high - low), 0.0, 1.0)
    return first_n, second_n, float(low), float(high)


def choose_serif_font() -> str:
    """Select an embeddable serif font and return its actual family name."""
    candidates = (
        "Times New Roman",
        "Liberation Serif",
        "TeX Gyre Termes",
        "Nimbus Roman",
        "STIX Two Text",
        "DejaVu Serif",
    )
    for family in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            return font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
    return "DejaVu Serif"


def configure_plot_style(figure_dpi: int) -> str:
    """Configure reproducible IEEE-style typography and rasterization."""
    selected_font = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": selected_font,
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": figure_dpi,
            "savefig.dpi": figure_dpi,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "image.composite_image": False,
        }
    )
    return selected_font


def draw_scalar_map(
    axis: mpl.axes.Axes,
    image_rgb: np.ndarray,
    scalar_map: np.ndarray,
    cmap: str,
    norm: Normalize,
    map_style: str,
    overlay_alpha: float,
) -> mpl.image.AxesImage:
    """Render a scalar map either faithfully or as a qualitative overlay."""
    if map_style == "overlay":
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        axis.imshow(
            image_gray,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="bilinear",
        )
        artist = axis.imshow(
            scalar_map,
            cmap=cmap,
            norm=norm,
            alpha=overlay_alpha,
            interpolation="bilinear",
        )
    else:
        artist = axis.imshow(
            scalar_map,
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
        )
    axis.set_axis_off()
    return artist


def pdf_raster_preflight(pdf_path: Path, min_ppi: int) -> dict[str, Any]:
    """Inspect PDF raster layers with pdfimages when Poppler is available."""
    executable = shutil.which("pdfimages")
    if executable is None:
        return {
            "tool": None,
            "status": "not_checked",
            "reason": "pdfimages was not found on PATH",
            "minimum_required_ppi": min_ppi,
        }

    completed = subprocess.run(
        [executable, "-list", str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "tool": executable,
            "status": "error",
            "reason": completed.stderr.strip(),
            "minimum_required_ppi": min_ppi,
        }

    layers: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 16 or not fields[0].isdigit() or fields[2] != "image":
            continue
        try:
            layers.append(
                {
                    "page": int(fields[0]),
                    "image_number": int(fields[1]),
                    "width_px": int(fields[3]),
                    "height_px": int(fields[4]),
                    "x_ppi": int(fields[12]),
                    "y_ppi": int(fields[13]),
                }
            )
        except (ValueError, IndexError):
            continue

    min_detected = min(
        (min(item["x_ppi"], item["y_ppi"]) for item in layers),
        default=None,
    )
    passed = min_detected is not None and min_detected >= min_ppi
    result = {
        "tool": executable,
        "status": "pass" if passed else "fail",
        "minimum_required_ppi": min_ppi,
        "minimum_detected_ppi": min_detected,
        "raster_layers": layers,
    }
    if not passed:
        warnings.warn(
            f"{pdf_path.name} contains a raster layer below {min_ppi} ppi: "
            f"minimum detected={min_detected}. Do not submit this PDF yet.",
            RuntimeWarning,
        )
    return result


def save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    figure_dpi: int,
    pdf_min_ppi: int,
) -> dict[str, Any]:
    """Save exact-size PNG/PDF files and run an optional PDF raster preflight."""
    figure.set_dpi(figure_dpi)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    save_kwargs = {
        "dpi": figure_dpi,
        "facecolor": "white",
        "edgecolor": "white",
        "transparent": False,
    }
    # Deliberately avoid bbox_inches='tight': it changes final physical size and PPI.
    figure.savefig(png_path, **save_kwargs)
    figure.savefig(pdf_path, **save_kwargs)
    width_in, height_in = figure.get_size_inches()
    return {
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "figure_size_in": [float(width_in), float(height_in)],
        "figure_dpi": figure_dpi,
        "pdf_raster_preflight": pdf_raster_preflight(pdf_path, pdf_min_ppi),
    }


def plot_apa(
    image_rgb: np.ndarray,
    display_weights: dict[int, np.ndarray],
    output_dir: Path,
    map_style: str,
    overlay_alpha: float,
    figure_dpi: int,
    pdf_min_ppi: int,
) -> dict[str, Any]:
    """Create a two-column-width input panel and 3x3 APA weight matrix."""
    targets = sorted(display_weights)
    if targets != [0, 1, 2]:
        raise RuntimeError(f"Expected APA targets [0,1,2], found {targets}.")

    selected_font = configure_plot_style(figure_dpi)
    figure = plt.figure(
        figsize=(7.16, 5.55),
        dpi=figure_dpi,
        layout="constrained",
    )
    figure.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    grid = figure.add_gridspec(
        4,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.035),
        height_ratios=(0.72, 1.0, 1.0, 1.0),
    )

    input_axis = figure.add_subplot(grid[0, :3])
    input_axis.imshow(image_rgb, interpolation="bilinear")
    input_axis.set_title("(a) Input image", pad=1.5)
    input_axis.set_axis_off()

    source_names = ("Source P2", "Source P3", "Source P4")
    target_names = ("Target PA-P2", "Target PA-P3", "Target PA-P4")
    norm = Normalize(vmin=0.0, vmax=1.0)
    last_artist = None
    panel_code = ord("b")
    sum_to_one_errors: dict[str, float] = {}

    for row, target in enumerate(targets, start=1):
        weights = display_weights[target]
        if weights.shape[0] != 3:
            raise RuntimeError(f"Target {target} has {weights.shape[0]} sources, expected 3.")
        sum_to_one_errors[f"target_{target}"] = float(
            np.max(np.abs(np.sum(weights, axis=0) - 1.0))
        )
        for column in range(3):
            axis = figure.add_subplot(grid[row, column])
            last_artist = draw_scalar_map(
                axis=axis,
                image_rgb=image_rgb,
                scalar_map=weights[column],
                cmap="viridis",
                norm=norm,
                map_style=map_style,
                overlay_alpha=overlay_alpha,
            )
            axis.set_title(f"({chr(panel_code)}) {source_names[column]}", pad=1.5)
            if column == 0:
                axis.text(
                    -0.028,
                    0.5,
                    target_names[row - 1],
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=8.0,
                )
            panel_code += 1

    if last_artist is None:
        raise RuntimeError("No APA maps were rendered.")
    color_axis = figure.add_subplot(grid[1:, 3])
    colorbar = figure.colorbar(last_artist, cax=color_axis)
    colorbar.set_label(r"Fusion weight $\alpha_k^l$")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))

    audit = save_figure(
        figure,
        output_dir,
        "apa_weights_from_best",
        figure_dpi,
        pdf_min_ppi,
    )
    plt.close(figure)
    audit.update(
        {
            "font_family": selected_font,
            "map_style": map_style,
            "max_sum_to_one_error": sum_to_one_errors,
        }
    )
    return audit


def plot_pgd(
    image_rgb: np.ndarray,
    gt_heat: np.ndarray | None,
    predicted_heat: np.ndarray,
    before_response: np.ndarray,
    after_response: np.ndarray,
    output_dir: Path,
    map_style: str,
    overlay_alpha: float,
    figure_dpi: int,
    pdf_min_ppi: int,
) -> tuple[dict[str, float | int | None], dict[str, Any]]:
    """Create a compact two-column PGD figure with scientifically shared scales."""
    before_n, after_n, feature_low, feature_high = normalize_joint(
        before_response, after_response
    )
    # R is the channel-RMS response (not a raw tensor feature F).
    delta = after_response - before_response
    delta_limit = float(np.percentile(np.abs(delta), 99.0))
    if not np.isfinite(delta_limit) or delta_limit <= 0:
        delta_limit = max(float(np.max(np.abs(delta))), 1e-6)

    selected_font = configure_plot_style(figure_dpi)
    figure = plt.figure(figsize=(7.16, 3.55), dpi=figure_dpi)
    outer = figure.add_gridspec(
        2,
        1,
        left=0.015,
        right=0.92,
        bottom=0.035,
        top=0.965,
        hspace=0.17,
    )
    top = outer[0].subgridspec(
        1, 4, width_ratios=(1.0, 1.0, 1.0, 0.035), wspace=0.05
    )
    bottom = outer[1].subgridspec(
        1, 6,
        width_ratios=(1.0, 1.0, 0.035, 0.12, 1.0, 0.035),
        wspace=0.05,
    )

    ax_a = figure.add_subplot(top[0, 0])
    ax_b = figure.add_subplot(top[0, 1])
    ax_c = figure.add_subplot(top[0, 2])
    cax_heat = figure.add_subplot(top[0, 3])
    ax_d = figure.add_subplot(bottom[0, 0])
    ax_e = figure.add_subplot(bottom[0, 1])
    cax_response = figure.add_subplot(bottom[0, 2])
    ax_f = figure.add_subplot(bottom[0, 4])
    cax_delta = figure.add_subplot(bottom[0, 5])

    heat_norm = Normalize(0.0, 1.0)
    feature_norm = Normalize(0.0, 1.0)
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    ax_a.imshow(image_rgb, interpolation="bilinear")
    ax_a.set_title("(a) Input image", pad=1.5)
    ax_a.set_axis_off()

    if gt_heat is not None:
        draw_scalar_map(
            ax_b, image_rgb, gt_heat, "viridis", heat_norm, map_style, overlay_alpha
        )
        ax_b.set_title("(b) Gaussian target", pad=1.5)
    else:
        ax_b.imshow(np.ones((*image_rgb.shape[:2], 3), dtype=np.float32))
        ax_b.text(
            0.5,
            0.5,
            "GT label not supplied",
            ha="center",
            va="center",
            transform=ax_b.transAxes,
            fontsize=8,
        )
        ax_b.set_title("(b) Target unavailable", pad=1.5)
        ax_b.set_axis_off()

    pred_artist = draw_scalar_map(
        ax_c,
        image_rgb,
        predicted_heat,
        "viridis",
        heat_norm,
        map_style,
        overlay_alpha,
    )
    ax_c.set_title("(c) Predicted heatmap", pad=1.5)
    heat_bar = figure.colorbar(pred_artist, cax=cax_heat)
    heat_bar.set_label("Heat value")
    heat_bar.set_ticks(np.linspace(0.0, 1.0, 6))

    before_artist = draw_scalar_map(
        ax_d, image_rgb, before_n, "viridis", feature_norm, map_style, overlay_alpha
    )
    ax_d.set_title("(d) Before PGD", pad=1.5)
    draw_scalar_map(
        ax_e, image_rgb, after_n, "viridis", feature_norm, map_style, overlay_alpha
    )
    ax_e.set_title("(e) After PGD", pad=1.5)
    response_bar = figure.colorbar(before_artist, cax=cax_response)
    response_bar.set_label("Normalized response", fontsize=7.0, labelpad=2.0)
    response_bar.set_ticks(np.linspace(0.0, 1.0, 6))

    delta_artist = draw_scalar_map(
        ax_f, image_rgb, delta, "coolwarm", delta_norm, map_style, overlay_alpha
    )
    ax_f.set_title(r"(f) Difference $\Delta R$", pad=1.5)
    delta_bar = figure.colorbar(delta_artist, cax=cax_delta)
    delta_bar.set_label(r"$\Delta R$ (a.u.)")

    audit = save_figure(
        figure,
        output_dir,
        "pgd_responses_from_best",
        figure_dpi,
        pdf_min_ppi,
    )
    plt.close(figure)
    audit.update({"font_family": selected_font, "map_style": map_style})

    metrics: dict[str, float | int | None] = {
        "feature_shared_percentile_low": feature_low,
        "feature_shared_percentile_high": feature_high,
        "delta_symmetric_limit": delta_limit,
        "gt_pred_pearson": None,
        "predicted_heat_foreground_background_ratio": None,
        "delta_foreground_mean": None,
        "delta_background_mean": None,
    }
    if gt_heat is not None:
        foreground = gt_heat >= 0.3
        background = gt_heat <= 0.05
        flat_gt = gt_heat.reshape(-1)
        flat_pred = predicted_heat.reshape(-1)
        if np.std(flat_gt) > 0 and np.std(flat_pred) > 0:
            metrics["gt_pred_pearson"] = float(np.corrcoef(flat_gt, flat_pred)[0, 1])
        if foreground.any() and background.any():
            fg_pred = float(predicted_heat[foreground].mean())
            bg_pred = float(predicted_heat[background].mean())
            metrics["predicted_heat_foreground_background_ratio"] = fg_pred / max(
                bg_pred, 1e-8
            )
            metrics["delta_foreground_mean"] = float(delta[foreground].mean())
            metrics["delta_background_mean"] = float(delta[background].mean())
    return metrics, audit

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "raw_tensors"
    display_dir = args.output / "display_maps"
    raw_dir.mkdir(exist_ok=True)
    display_dir.mkdir(exist_ok=True)

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_hw = image_rgb.shape[:2]
    cv2.imwrite(str(args.output / "input_image.png"), image_bgr)

    detector = YOLO(str(args.weights))
    network = detector.model
    network.eval()

    pa_modules = [
        (name, module)
        for name, module in network.named_modules()
        if module.__class__.__name__ == "PAFeature"
    ]
    pgd_modules = [
        (name, module)
        for name, module in network.named_modules()
        if module.__class__.__name__ == "PGDHeatGate"
    ]
    if len(pa_modules) != 3:
        raise RuntimeError(
            f"Expected three PAFeature modules, found {len(pa_modules)}: "
            f"{[name for name, _ in pa_modules]}"
        )
    if not pgd_modules:
        raise RuntimeError("No PGDHeatGate module was found in the checkpoint.")

    handles: list[Any] = []
    model_input_hw: dict[str, tuple[int, int]] = {}
    pa_cache: dict[int, dict[str, Any]] = {}
    pgd_cache: dict[str, dict[str, Any]] = {}

    def root_pre_hook(_module: Any, inputs: Any) -> None:
        tensor = first_tensor(inputs)
        if tensor.ndim == 4:
            model_input_hw["value"] = (int(tensor.shape[-2]), int(tensor.shape[-1]))

    handles.append(network.register_forward_pre_hook(root_pre_hook))

    for module_name, pa_module in pa_modules:
        target = int(pa_module.target)
        nl = int(pa_module.nl)

        def pa_weight_hook(
            _weight_module: Any,
            inputs: Any,
            output: Any,
            *,
            target_index: int = target,
            source_count: int = nl,
            parent_name: str = module_name,
        ) -> None:
            concatenated = first_tensor(inputs)
            logits = first_tensor(output)
            if logits.ndim != 4 or logits.shape[1] != source_count:
                raise RuntimeError(
                    f"{parent_name}.weight produced {tuple(logits.shape)}, "
                    f"expected Bx{source_count}xHxW logits."
                )
            if concatenated.shape[1] % source_count != 0:
                raise RuntimeError(
                    f"Cannot split {parent_name}.weight input channels "
                    f"{concatenated.shape[1]} into {source_count} sources."
                )
            weights = torch.softmax(logits.float(), dim=1)
            aligned_sources = torch.chunk(concatenated.float(), source_count, dim=1)
            source_responses = torch.stack(
                [
                    channel_rms(source) * weights[:, source_index]
                    for source_index, source in enumerate(aligned_sources)
                ],
                dim=1,
            )
            pa_cache[target_index] = {
                "module": parent_name,
                "weights": weights[0].detach().cpu().numpy().astype(np.float32),
                "responses": source_responses[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            }

        handles.append(pa_module.weight.register_forward_hook(pa_weight_hook))

    for module_name, pgd_module in pgd_modules:
        key = module_name
        pgd_cache[key] = {
            "module": module_name,
            "stride": int(pgd_module.stride_level),
            "gate": float(pgd_module.gate),
        }
        if hasattr(pgd_module, "save_heat"):
            pgd_module.save_heat = True

        def pgd_before_hook(
            _module: Any, _inputs: Any, output: Any, *, cache_key: str = key
        ) -> None:
            tensor = first_tensor(output)
            pgd_cache[cache_key]["before"] = (
                channel_rms(tensor)[0].detach().cpu().numpy().astype(np.float32)
            )

        def pgd_heat_hook(
            _module: Any, _inputs: Any, output: Any, *, cache_key: str = key
        ) -> None:
            logits = first_tensor(output)
            heat = torch.sigmoid(logits.float())
            pgd_cache[cache_key]["heat"] = (
                heat[0, 0].detach().cpu().numpy().astype(np.float32)
            )

        def pgd_after_hook(
            _module: Any, _inputs: Any, output: Any, *, cache_key: str = key
        ) -> None:
            tensor = first_tensor(output)
            pgd_cache[cache_key]["after"] = (
                channel_rms(tensor)[0].detach().cpu().numpy().astype(np.float32)
            )

        handles.append(pgd_module.proj.register_forward_hook(pgd_before_hook))
        handles.append(pgd_module.heat_head.register_forward_hook(pgd_heat_hook))
        handles.append(pgd_module.register_forward_hook(pgd_after_hook))

    try:
        detector.predict(
            source=str(args.image),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            augment=False,
            save=False,
            verbose=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    if "value" not in model_input_hw:
        raise RuntimeError("The model-input size was not captured.")
    model_hw = model_input_hw["value"]

    display_weights: dict[int, np.ndarray] = {}
    for target in sorted(pa_cache):
        weights = pa_cache[target]["weights"]
        responses = pa_cache[target]["responses"]
        np.save(raw_dir / f"apa_target_{target}_weights.npy", weights)
        np.save(raw_dir / f"apa_target_{target}_responses.npy", responses)
        display_weight = np.stack(
            [map_to_original(item, model_hw, original_hw) for item in weights], axis=0
        )
        display_response = np.stack(
            [map_to_original(item, model_hw, original_hw) for item in responses], axis=0
        )
        display_weights[target] = display_weight
        np.save(display_dir / f"apa_target_{target}_weights_display.npy", display_weight)
        np.save(
            display_dir / f"apa_target_{target}_responses_display.npy", display_response
        )

    active_pgd = [
        item
        for item in pgd_cache.values()
        if item.get("gate", 0.0) > 0.0
        and all(name in item for name in ("heat", "before", "after"))
    ]
    if not active_pgd:
        raise RuntimeError(
            "No active PGD gate produced heat, before, and after tensors. "
            f"Captured metadata: {pgd_cache}"
        )
    if args.pgd_stride is not None:
        matching_pgd = [
            item for item in active_pgd if int(item["stride"]) == args.pgd_stride
        ]
        if len(matching_pgd) != 1:
            available = sorted(int(item["stride"]) for item in active_pgd)
            raise RuntimeError(
                f"--pgd-stride={args.pgd_stride} matched {len(matching_pgd)} active "
                f"modules; available strides/levels are {available}."
            )
        selected_pgd = matching_pgd[0]
    elif len(active_pgd) == 1:
        selected_pgd = active_pgd[0]
    else:
        available = sorted(int(item["stride"]) for item in active_pgd)
        raise RuntimeError(
            "Multiple active PGDHeatGate modules were captured. Pass --pgd-stride "
            f"explicitly to avoid silent/cherry-picked selection. Available: {available}."
        )
    predicted_heat_native = selected_pgd["heat"]
    before_native = selected_pgd["before"]
    after_native = selected_pgd["after"]
    np.save(raw_dir / "pgd_predicted_heat.npy", predicted_heat_native)
    np.save(raw_dir / "pgd_before_response.npy", before_native)
    np.save(raw_dir / "pgd_after_response.npy", after_native)
    np.save(raw_dir / "pgd_response_difference.npy", after_native - before_native)

    predicted_heat = map_to_original(predicted_heat_native, model_hw, original_hw)
    before_response = map_to_original(before_native, model_hw, original_hw)
    after_response = map_to_original(after_native, model_hw, original_hw)
    np.save(display_dir / "pgd_predicted_heat_display.npy", predicted_heat)
    np.save(display_dir / "pgd_before_response_display.npy", before_response)
    np.save(display_dir / "pgd_after_response_display.npy", after_response)
    np.save(
        display_dir / "pgd_response_difference_display.npy",
        after_response - before_response,
    )

    gt_heat_native: np.ndarray | None = None
    gt_heat_display: np.ndarray | None = None
    retained_objects: int | None = None
    if args.label is not None:
        if not args.label.exists():
            raise FileNotFoundError(f"Could not read label file: {args.label}")
        gt_heat_native, retained_objects = build_gaussian_target(
            label_path=args.label,
            original_hw=original_hw,
            model_hw=model_hw,
            heat_hw=predicted_heat_native.shape,
            small_area=args.small_area,
            eta_bins=tuple(args.eta_bins),
            small_classes=tuple(args.small_classes),
        )
        gt_heat_display = map_to_original(gt_heat_native, model_hw, original_hw)
        np.save(raw_dir / "pgd_gt_heat.npy", gt_heat_native)
        np.save(display_dir / "pgd_gt_heat_display.npy", gt_heat_display)

    apa_figure_audit = plot_apa(
        image_rgb=image_rgb,
        display_weights=display_weights,
        output_dir=args.output,
        map_style=args.map_style,
        overlay_alpha=args.overlay_alpha,
        figure_dpi=args.figure_dpi,
        pdf_min_ppi=args.pdf_min_ppi,
    )
    pgd_metrics, pgd_figure_audit = plot_pgd(
        image_rgb=image_rgb,
        gt_heat=gt_heat_display,
        predicted_heat=predicted_heat,
        before_response=before_response,
        after_response=after_response,
        output_dir=args.output,
        map_style=args.map_style,
        overlay_alpha=args.overlay_alpha,
        figure_dpi=args.figure_dpi,
        pdf_min_ppi=args.pdf_min_ppi,
    )

    metadata = {
        "weights": str(args.weights.resolve()),
        "image": str(args.image.resolve()),
        "label": str(args.label.resolve()) if args.label is not None else None,
        "original_hw": list(original_hw),
        "model_input_hw": list(model_hw),
        "imgsz_argument": args.imgsz,
        "apa_modules": [
            {
                "name": name,
                "target": int(module.target),
                "sources": int(module.nl),
            }
            for name, module in pa_modules
        ],
        "pgd_modules": [
            {
                "name": name,
                "stride": int(module.stride_level),
                "gate": float(module.gate),
                "loss_weight": float(module.loss_weight),
            }
            for name, module in pgd_modules
        ],
        "selected_pgd": {
            "module": selected_pgd["module"],
            "stride": int(selected_pgd["stride"]),
            "gate": float(selected_pgd["gate"]),
        },
        "small_area": args.small_area,
        "small_classes": list(args.small_classes),
        "eta_bins": list(args.eta_bins),
        "retained_gt_objects": retained_objects,
        "map_style": args.map_style,
        "figure_dpi": args.figure_dpi,
        "pdf_min_ppi": args.pdf_min_ppi,
        "apa_figure_audit": apa_figure_audit,
        "pgd_figure_audit": pgd_figure_audit,
        "pgd_metrics": pgd_metrics,
        "notes": [
            "APA weight maps are the branch-wise softmax of PAFeature.weight logits.",
            "APA source-response maps are channel-RMS aligned features multiplied by their source weights.",
            "PGD before/after maps use channel RMS and one shared robust normalization range.",
            "Delta R is the difference between unnormalized channel-RMS response maps and is reported in arbitrary units.",
            "Raw-map rendering is preferred for main-paper quantitative figures; overlay rendering is qualitative.",
            "The GT panel is valid only if small_area, small_classes, eta_bins, and letterbox geometry exactly match the training target builder.",
        ],
    }
    (args.output / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved APA and PGD visualizations to: {args.output.resolve()}")
    print(json.dumps(metadata["selected_pgd"], indent=2))


if __name__ == "__main__":
    main()
