import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from lib.utils import preprocess_filename


RooflineParam = tuple[float, float]
RooflineData = tuple[np.ndarray, np.ndarray, float, float]


__all__ = ["plot_roofline", "plot_roofline_from_df", "plot_roofline_from_csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--example", action="store_true", help="run the example code")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to the input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="path to the output figure, None for the same directory of the input file",
    )
    return parser.parse_args()


def get_roofline(
    peak_performance: float, peak_bandwidth: float, max_op_intensity: float = 30
) -> RooflineData:
    intensity = np.linspace(0, max_op_intensity, 200)
    compute_roof = np.full_like(intensity, peak_performance)
    bandwidth_roof = intensity * peak_bandwidth
    roofline = np.minimum(compute_roof, bandwidth_roof)
    return intensity, roofline, peak_performance, peak_bandwidth


def plot_roofline(
    rooflines: dict[str, RooflineParam],
    workloads: dict[str, float] | None = None,
    filename: str | Path = "log/figure/roofline.png",
) -> None:
    plt.figure(figsize=(8, 6))
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    oi_max = 0.0

    # Plot workload intensities
    if workloads is not None:
        colors = [plt.get_cmap("tab10")(i) for i in range(len(workloads))]
        for (k, v), color in zip(workloads.items(), colors):
            plt.axvline(x=v, color=color, linestyle="--", label=f"{k} (OI = {v:.2f})")
            oi_max = max(oi_max, v)

    # Plot rooflines
    for i, (key, (perf, band)) in enumerate(rooflines.items()):
        x, y, *_ = get_roofline(perf, band, max_op_intensity=oi_max * 1.05)
        color = "black" if i == len(rooflines) - 1 else "#aaaaaa"
        plt.plot(x, y, linewidth=2, color=color, label=key)
        xmin = min(xmin, x[0])
        xmax = max(xmax, x[-1])
        ymin = min(ymin, y[0])
        ymax = max(ymax, y[-1])

    # Plot settings
    plt.xlabel("Operational Intensity (MACs/byte)")
    plt.ylabel("Performance (MACs/cycle)")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax * 1.05)
    plt.title("Roofline Model")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Save figure
    path = Path(preprocess_filename(filename, existed="overwrite"))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f"Roofline plot saved at {path}")


def plot_roofline_from_df(df: pd.DataFrame, ofile: str | Path) -> None:
    # Load the roofline and workload data
    roofline_params = frozenset(zip(df["peak_performance"], df["peak_bandwidth"]))
    rooflines = {f"Roofline of Hardware {i}": v for i, v in enumerate(roofline_params)}
    workloads = {k: v for k, v in zip(df["layer"], df["intensity"])}
    print(f"{len(rooflines)} rooflines and {len(workloads)} workloads loaded.")

    # Plot the roofline model
    plot_roofline(rooflines, workloads, ofile)


def plot_roofline_from_csv(ifile: str | Path, ofile: str | Path) -> None:
    df = pd.read_csv(ifile)
    plot_roofline_from_df(df, ofile)


def plot_example():
    plot_roofline(
        rooflines={"Roofline": (48.0, 4.0)},
        workloads={"Machine balance point": 12.0},
        filename="../log/figure/baseline.png",
    )
    plot_roofline(
        rooflines={"Roofline": (48.0, 4.0)},
        workloads={"Mapping 1": 8.0, "Mapping 2": 18.0},
        filename="../log/figure/mappings.png",
    )
    plot_roofline(
        rooflines={
            "Hardware 1 (48 PEs)": (48.0, 4.0),
            "Hardware 2 (72 PEs)": (72.0, 4.0),
        },
        workloads={"Workload": 16.0},
        filename="../log/figure/hardware.png",
    )


def main() -> None:
    args = parse_args()

    if args.example:
        plot_example()
        return

    if args.output is None:
        args.output = args.input.replace(".csv", ".png")

    plot_roofline_from_csv(args.input, args.output)


if __name__ == "__main__":
    main()
