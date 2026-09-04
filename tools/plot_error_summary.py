#!/usr/bin/env python3
"""Generate a recording-wide sampled ERD-versus-EDF error summary figure.

The comparison is deliberately split into two populations:

* samples whose native ERD digital value is representable by EDF int16;
* all finite samples, including values clipped during EDF export.

Windows are distributed uniformly over the common ERD/EDF duration.  Mean,
absolute-mean, RMS, standard deviation, and extrema are exact for every sample
inside those windows.  Quantiles use a deterministic stride subsample to keep
memory bounded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from natus_erd import EDFReader, NatusERDReader


SIGNAL_CHANNEL_COUNT = 256
COLORS = {
    "mae": "#0072B2",
    "rmse": "#D55E00",
    "clip": "#CC79A7",
    "distribution": "#56B4E9",
    "channel": "#009E73",
    "reference": "#5B6573",
    "text": "#17212B",
    "muted": "#677381",
    "grid": "#D8DEE5",
}


@dataclass
class Moments:
    count: int = 0
    total: float = 0.0
    absolute_total: float = 0.0
    square_total: float = 0.0
    maximum_absolute: float = 0.0

    def add(self, values: np.ndarray[Any, Any]) -> None:
        if values.size == 0:
            return
        values64 = np.asarray(values, dtype=np.float64)
        self.count += int(values64.size)
        self.total += float(np.sum(values64, dtype=np.float64))
        self.absolute_total += float(np.sum(np.abs(values64), dtype=np.float64))
        self.square_total += float(
            np.sum(np.square(values64), dtype=np.float64)
        )
        self.maximum_absolute = max(
            self.maximum_absolute, float(np.max(np.abs(values64)))
        )

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "bias_uv": None,
                "mae_uv": None,
                "rmse_uv": None,
                "std_uv": None,
                "max_abs_uv": None,
            }
        bias = self.total / self.count
        mean_square = self.square_total / self.count
        return {
            "count": self.count,
            "bias_uv": bias,
            "mae_uv": self.absolute_total / self.count,
            "rmse_uv": math.sqrt(mean_square),
            "std_uv": math.sqrt(max(0.0, mean_square - bias * bias)),
            "max_abs_uv": self.maximum_absolute,
        }


def discover_edf(recording: Path) -> Path:
    resolved = recording.expanduser().resolve(strict=True)
    bases = [resolved if resolved.is_dir() else resolved.parent]
    if bases[0].parent not in bases:
        bases.append(bases[0].parent)
    for base in bases:
        candidates = sorted(
            candidate.resolve()
            for candidate in base.rglob("*.edf")
            if not any(part.casefold() == "decimated" for part in candidate.parts)
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError("Found multiple EDF files; pass --edf explicitly")
    raise FileNotFoundError("No EDF file found; pass --edf explicitly")


def finite_quantiles(
    values: np.ndarray[Any, Any], probabilities: tuple[float, ...]
) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"p{probability * 100:g}_abs_uv": None for probability in probabilities}
    absolute = np.abs(finite)
    return {
        f"p{probability * 100:g}_abs_uv": float(np.quantile(absolute, probability))
        for probability in probabilities
    }


def metric_text(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    absolute = abs(numeric)
    if absolute != 0 and (absolute < 0.001 or absolute >= 10_000):
        return f"{numeric:.3e}"
    return f"{numeric:.{digits}f}"


def select_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def analyse(
    recording: Path,
    edf_path: Path | None,
    *,
    window_count: int,
    window_seconds: float,
    quantile_stride: int,
) -> dict[str, Any]:
    erd = NatusERDReader.open(recording)
    edf = EDFReader(edf_path or discover_edf(recording))
    sample_rate = erd.info.sample_rate
    if not math.isclose(sample_rate, edf.sample_rate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ERD and EDF sample rates do not match")

    channels = tuple(
        index
        for index in range(SIGNAL_CHANNEL_COUNT)
        if not erd.channels[index].shorted
        and erd.channels[index].scale_uv_per_count is not None
    )
    shorted = tuple(
        index for index in range(SIGNAL_CHANNEL_COUNT) if erd.channels[index].shorted
    )
    if not channels:
        raise ValueError("No calibrated, non-shorted signal channels found")
    if len(edf.signals) < SIGNAL_CHANNEL_COUNT:
        raise ValueError("EDF contains fewer than 256 signal channels")
    for channel in channels:
        signal = edf.signals[channel]
        if not math.isclose(signal.sample_rate, sample_rate, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"EDF channel {channel} has a different sample rate")
        if signal.label != erd.channels[channel].name:
            raise ValueError(f"ERD/EDF channel labels differ at index {channel}")
        if signal.unit.replace("µ", "u").replace("μ", "u").lower() != "uv":
            raise ValueError(f"EDF channel {channel} is not calibrated in microvolts")

    common_samples = min(erd.info.n_samples, edf.n_samples)
    window_samples = max(1, round(window_seconds * sample_rate))
    if window_samples > common_samples:
        raise ValueError("Requested window is longer than the common recording duration")
    starts = np.unique(
        np.linspace(
            0,
            common_samples - window_samples,
            window_count,
            dtype=np.int64,
        )
    )
    scales = np.asarray(
        [erd.channels[index].scale_uv_per_count for index in channels],
        dtype=np.float64,
    )

    in_range_moments = Moments()
    all_moments = Moments()
    clipped_moments = Moments()
    digital_moments = Moments()
    quantile_in_range: list[np.ndarray[Any, Any]] = []
    quantile_all: list[np.ndarray[Any, Any]] = []

    channel_count = np.zeros(len(channels), dtype=np.int64)
    channel_square_sum = np.zeros(len(channels), dtype=np.float64)
    channel_absolute_sum = np.zeros(len(channels), dtype=np.float64)

    window_rows: list[dict[str, float | int]] = []
    for ordinal, start in enumerate(starts, start=1):
        start_sample = int(start)
        stop = start_sample + window_samples
        erd_digital = erd.read_samples(
            start_sample, stop, channels, units="digital"
        )
        edf_digital = edf.read_digital(start_sample, stop, channels)
        erd_uv = erd_digital * scales[:, np.newaxis]
        edf_uv = edf.digital_to_physical(edf_digital, channels)
        residual = erd_uv - edf_uv

        finite = np.isfinite(residual) & np.isfinite(erd_digital)
        in_range = finite & (erd_digital >= -32768) & (erd_digital <= 32767)
        clipped = finite & ~in_range

        in_values = residual[in_range]
        all_values = residual[finite]
        clipped_values = residual[clipped]
        digital_values = (
            erd_digital[in_range] - edf_digital.astype(np.float64)[in_range]
        )

        in_range_moments.add(in_values)
        all_moments.add(all_values)
        clipped_moments.add(clipped_values)
        digital_moments.add(digital_values)
        quantile_in_range.append(in_values[::quantile_stride].copy())
        quantile_all.append(all_values[::quantile_stride].copy())

        safe_residual = np.where(in_range, residual, 0.0)
        channel_count += np.count_nonzero(in_range, axis=1)
        channel_square_sum += np.sum(np.square(safe_residual), axis=1)
        channel_absolute_sum += np.sum(np.abs(safe_residual), axis=1)

        in_summary = Moments()
        in_summary.add(in_values)
        summary = in_summary.summary()
        finite_count = int(np.count_nonzero(finite))
        clipped_count = int(np.count_nonzero(clipped))
        window_rows.append(
            {
                "window": ordinal,
                "start_sample": start_sample,
                "time_hours": (start_sample + window_samples / 2) / sample_rate / 3600,
                "finite_samples": finite_count,
                "in_range_samples": int(np.count_nonzero(in_range)),
                "clipped_samples": clipped_count,
                "clipped_rate": clipped_count / finite_count if finite_count else 0.0,
                "bias_uv": float(summary["bias_uv"] or 0.0),
                "mae_uv": float(summary["mae_uv"] or 0.0),
                "rmse_uv": float(summary["rmse_uv"] or 0.0),
                "max_abs_uv": float(summary["max_abs_uv"] or 0.0),
            }
        )
        if ordinal == 1 or ordinal % 4 == 0 or ordinal == len(starts):
            print(f"analysed window {ordinal}/{len(starts)}", flush=True)

    in_sample = np.concatenate(quantile_in_range)
    all_sample = np.concatenate(quantile_all)
    in_summary = in_range_moments.summary()
    in_summary.update(finite_quantiles(in_sample, (0.5, 0.95, 0.99, 0.999)))
    all_summary = all_moments.summary()
    all_summary.update(finite_quantiles(all_sample, (0.5, 0.95, 0.99, 0.999)))

    with np.errstate(invalid="ignore", divide="ignore"):
        channel_rmse = np.sqrt(channel_square_sum / channel_count)
        channel_mae = channel_absolute_sum / channel_count

    total_finite = int(all_moments.count)
    clipped_count = int(clipped_moments.count)
    covered_seconds = len(starts) * window_samples / sample_rate
    result: dict[str, Any] = {
        "definition": "ERD - EDF",
        "units": "µV",
        "sampling": {
            "method": "uniform stratified windows over common recording duration",
            "window_count": len(starts),
            "window_seconds": window_samples / sample_rate,
            "covered_seconds": covered_seconds,
            "recording_duration_seconds": common_samples / sample_rate,
            "time_coverage_fraction": covered_seconds / (common_samples / sample_rate),
            "channels": len(channels),
            "shorted_channels_excluded": list(shorted),
            "channel_samples": total_finite,
            "quantile_stride": quantile_stride,
            "quantile_sample_count": int(in_sample.size),
        },
        "in_edf_int16_range": in_summary,
        "all_finite_samples": all_summary,
        "edf_clipped_samples": {
            **clipped_moments.summary(),
            "rate": clipped_count / total_finite if total_finite else 0.0,
        },
        "digital_error_in_range": {
            "count": digital_moments.count,
            "bias_lsb": digital_moments.total / digital_moments.count
            if digital_moments.count
            else None,
            "mae_lsb": digital_moments.absolute_total / digital_moments.count
            if digital_moments.count
            else None,
            "rmse_lsb": math.sqrt(
                digital_moments.square_total / digital_moments.count
            )
            if digital_moments.count
            else None,
            "max_abs_lsb": digital_moments.maximum_absolute
            if digital_moments.count
            else None,
        },
        "windows": window_rows,
        "channels": [
            {
                "index": channel,
                "count": int(channel_count[row]),
                "mae_uv": float(channel_mae[row]),
                "rmse_uv": float(channel_rmse[row]),
            }
            for row, channel in enumerate(channels)
        ],
        "_plot_samples": {
            "in_range_residual": in_sample,
            "channel_indices": np.asarray(channels, dtype=np.int64),
            "channel_rmse": channel_rmse,
        },
    }
    return result


def make_figure(result: dict[str, Any], output_png: Path, output_pdf: Path) -> None:
    font = select_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.75,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
        }
    )

    figure = plt.figure(figsize=(13.6, 10.2), facecolor="#F7F9FB")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.15, 1.0, 0.82),
        left=0.075,
        right=0.95,
        top=0.90,
        bottom=0.07,
        hspace=0.38,
        wspace=0.26,
    )

    windows = result["windows"]
    times = np.asarray([row["time_hours"] for row in windows], dtype=np.float64)
    window_mae = np.asarray([row["mae_uv"] for row in windows], dtype=np.float64)
    window_rmse = np.asarray([row["rmse_uv"] for row in windows], dtype=np.float64)
    clip_ppm = np.asarray(
        [row["clipped_rate"] * 1_000_000 for row in windows], dtype=np.float64
    )

    trend = figure.add_subplot(grid[0, :])
    trend.set_facecolor("white")
    trend.plot(
        times,
        window_mae,
        color=COLORS["mae"],
        marker="o",
        markersize=3.8,
        linewidth=1.7,
        label="MAE（EDF 可表示范围内）",
    )
    trend.plot(
        times,
        window_rmse,
        color=COLORS["rmse"],
        marker="s",
        markersize=3.5,
        linewidth=1.7,
        label="RMSE（EDF 可表示范围内）",
    )
    trend.set_title("A  全记录均匀取窗：误差随记录时间的变化", loc="left")
    trend.set_xlabel("记录时间（小时）")
    trend.set_ylabel("误差（µV）")
    trend.legend(loc="upper left", ncol=2)
    trend.margins(x=0.01)

    clipping_axis = trend.twinx()
    clipping_axis.spines["top"].set_visible(False)
    clipping_axis.plot(
        times,
        clip_ppm,
        color=COLORS["clip"],
        marker="^",
        markersize=3.2,
        linewidth=1.1,
        alpha=0.72,
        label="EDF 削顶率",
    )
    clipping_axis.set_ylabel("削顶样本率（ppm）", color=COLORS["clip"])
    clipping_axis.tick_params(axis="y", colors=COLORS["clip"])
    clipping_axis.grid(False)

    distribution = figure.add_subplot(grid[1, 0])
    distribution.set_facecolor("white")
    residual = np.asarray(
        result["_plot_samples"]["in_range_residual"], dtype=np.float64
    )
    lower, upper = np.quantile(residual, [0.001, 0.999])
    if math.isclose(float(lower), float(upper)):
        lower, upper = float(lower) - 0.5, float(upper) + 0.5
    central = residual[(residual >= lower) & (residual <= upper)]
    distribution.hist(
        central,
        bins=90,
        color=COLORS["distribution"],
        edgecolor="white",
        linewidth=0.25,
        log=True,
    )
    bias = float(result["in_edf_int16_range"]["bias_uv"])
    distribution.axvline(0, color=COLORS["reference"], linestyle="--", linewidth=1)
    distribution.axvline(
        bias,
        color=COLORS["rmse"],
        linestyle="-",
        linewidth=1.3,
        label=f"Bias = {bias:.4f} µV",
    )
    distribution.set_title("B  有符号误差分布（中央 99.8%）", loc="left")
    distribution.set_xlabel("ERD − EDF（µV）")
    distribution.set_ylabel("样本数（对数刻度）")
    distribution.legend(loc="upper right")

    channel_axis = figure.add_subplot(grid[1, 1])
    channel_axis.set_facecolor("white")
    channel_indices = np.asarray(
        result["_plot_samples"]["channel_indices"], dtype=np.int64
    )
    channel_rmse = np.asarray(
        result["_plot_samples"]["channel_rmse"], dtype=np.float64
    )
    channel_axis.plot(
        channel_indices,
        channel_rmse,
        color=COLORS["channel"],
        linewidth=1.15,
    )
    channel_axis.fill_between(
        channel_indices,
        channel_rmse,
        color=COLORS["channel"],
        alpha=0.10,
    )
    for shorted in result["sampling"]["shorted_channels_excluded"]:
        channel_axis.axvline(
            shorted, color=COLORS["clip"], alpha=0.35, linewidth=0.8
        )
    channel_axis.set_title("C  各通道 RMSE（已排除 shorted 通道）", loc="left")
    channel_axis.set_xlabel("信号通道索引")
    channel_axis.set_ylabel("RMSE（µV）")
    channel_axis.set_xlim(-2, 257)

    table_axis = figure.add_subplot(grid[2, :])
    table_axis.axis("off")
    in_range = result["in_edf_int16_range"]
    all_finite = result["all_finite_samples"]
    rows = []
    for label, summary in (
        ("EDF 可表示范围内", in_range),
        ("全部有限样本（含削顶）", all_finite),
    ):
        rows.append(
            [
                label,
                f"{int(summary['count']):,}",
                metric_text(summary["bias_uv"]),
                metric_text(summary["mae_uv"]),
                metric_text(summary["rmse_uv"]),
                metric_text(summary["std_uv"]),
                metric_text(summary["p95_abs_uv"]),
                metric_text(summary["p99_abs_uv"]),
                metric_text(summary["max_abs_uv"]),
            ]
        )
    columns = [
        "统计范围",
        "通道-样本数",
        "Bias (µV)",
        "MAE (µV)",
        "RMSE (µV)",
        "Std (µV)",
        "P95 |误差|",
        "P99 |误差|",
        "Max |误差|",
    ]
    table = table_axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.28, 1.0, 0.65],
        colWidths=[0.17, 0.13, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE2E8")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#E9EFF5")
            cell.set_text_props(weight="bold", color=COLORS["text"])
        elif row == 1:
            cell.set_facecolor("#F4F9FC")
        else:
            cell.set_facecolor("#FFF7F2")
    table_axis.text(
        0.0,
        1.02,
        "D  总体误差统计",
        transform=table_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=COLORS["text"],
    )
    sampling = result["sampling"]
    clipping = result["edf_clipped_samples"]
    digital = result["digital_error_in_range"]
    note = (
        f"分层抽样：{sampling['window_count']} 个 × {sampling['window_seconds']:.1f} 秒窗口，"
        f"覆盖整段 {sampling['recording_duration_seconds'] / 3600:.3f} 小时；"
        f"{sampling['channels']} 个有效信号通道，共 {sampling['channel_samples']:,} 个通道-样本。  "
        f"EDF 削顶：{clipping['count']:,} 个（{clipping['rate'] * 1e6:.2f} ppm）。  "
        f"可表示范围内最大数字误差：{metric_text(digital['max_abs_lsb'], 2)} LSB。  "
        f"误差定义：ERD − EDF；分位数基于每 {sampling['quantile_stride']} 个样本取 1 个的确定性子样本。"
    )
    table_axis.text(
        0.0,
        0.14,
        note,
        transform=table_axis.transAxes,
        color=COLORS["muted"],
        fontsize=8.7,
        va="top",
        wrap=True,
    )

    figure.suptitle(
        "Natus ERD 与 EDF 导出误差总览",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
    )
    figure.text(
        0.075,
        0.93,
        "误差定义为 ERD − EDF；将格式换算误差与 EDF int16 削顶失真分开统计",
        color=COLORS["muted"],
        fontsize=10,
    )
    figure.savefig(output_png, facecolor=figure.get_facecolor())
    figure.savefig(output_pdf, facecolor=figure.get_facecolor())
    plt.close(figure)


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    plot_samples = result.pop("_plot_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / "erd_edf_error_statistics.json"
    statistics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    windows_path = output_dir / "erd_edf_window_statistics.csv"
    with windows_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["windows"][0]))
        writer.writeheader()
        writer.writerows(result["windows"])
    result["_plot_samples"] = plot_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--edf", type=Path)
    parser.add_argument("--windows", type=int, default=48)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--quantile-stride", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.windows < 5:
        parser.error("--windows must be at least 5")
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if args.quantile_stride < 1:
        parser.error("--quantile-stride must be at least 1")

    result = analyse(
        args.recording,
        args.edf,
        window_count=args.windows,
        window_seconds=args.window_seconds,
        quantile_stride=args.quantile_stride,
    )
    output_dir = args.output_dir.resolve()
    png = output_dir / "fig_erd_edf_error_summary.png"
    pdf = output_dir / "fig_erd_edf_error_summary.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(result, png, pdf)
    write_outputs(result, output_dir)
    print(f"saved {png}")
    print(f"saved {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
