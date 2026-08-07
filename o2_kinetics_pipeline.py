"""
O2 evolution kinetics pipeline for Hansatech O2view CSV exports.

For each input file: loads the trace, computes a smoothed O2 curve and its
instantaneous rate, auto-detects uphill (photosynthesis) / downhill
(respiration) phases, fits a rate to a fixed window centered on each phase
(avoiding edge/lag artifacts at the phase boundaries), and plots a 3-panel
figure (raw kinetic, instantaneous rate, segmented phases). A final combined
overlay plot compares all files' traces on one figure.

Usage:
    python o2_kinetics_pipeline.py file1.csv file2.csv file3.csv ...

Or in Jupyter/IPython, edit INPUT_FILES near the bottom and run the cell.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito colourblind-safe palette (distinguishable under deuteranopia,
# protanopia, and tritanopia -- deliberately avoids a red/green pairing).
COLORS = {"uphill": "#0072B2", "downhill": "#D55E00", "raw": "#999999",
          "smooth": "#E69F00", "rate": "#CC79A7"}
OVERLAY_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]


def load_o2_trace(file_path):
    """
    Load Time/Oxygen 1 from a Hansatech O2view CSV export.

    Header row is auto-detected (skiprows varies between exports: 29, 30,
    32 have all been seen) by scanning for the literal 'Time,Oxygen 1'
    column header -- deliberately NOT just 'Time,Oxygen', because some
    exports have an earlier decoy line ('TIME,OXYGEN,RATE,EVENT MARKERS')
    that would otherwise match first and give the wrong row.
    """
    with open(file_path, "r", errors="ignore") as f:
        header = next((i for i, line in enumerate(f)
                        if line.strip().lower().startswith("time,oxygen 1")), None)
    if header is None:
        raise ValueError(f"No 'Time,Oxygen 1' header found in {file_path}")

    # index_col=False guards against a known quirk where the header row has
    # one fewer field than the data rows (trailing comma), which otherwise
    # makes pandas silently shift every column by one.
    df = pd.read_csv(file_path, skiprows=header, index_col=False)
    df.columns = df.columns.str.strip()

    # pd.to_numeric(errors="coerce") -- NOT .astype(float) -- because every
    # O2view export ends with non-numeric footer rows ('End Of File',
    # 'Checksum, -123456'). .dropna() alone does not remove these: they
    # aren't NaN, they're literal text, so a bare .astype(float) crashes on
    # them. to_numeric+coerce turns them into NaN first, THEN dropna
    # removes them cleanly.
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Oxygen 1"] = pd.to_numeric(df["Oxygen 1"], errors="coerce")
    df = df.dropna(subset=["Time", "Oxygen 1"]).sort_values("Time").reset_index(drop=True)

    return df["Time"].values, df["Oxygen 1"].values


def process_and_plot(file_path, window_seconds=31, polyorder=3, min_duration=45,
                      fit_window_sec=30, save=True, show=False):
    """
    Full single-file pipeline: load, smooth, compute rate, detect
    uphill/downhill phases, fit a rate to a centered window within each
    phase, plot, and save.

    fit_window_sec: rather than fitting the rate across a whole detected
    phase (whose exact start/end is subject to a few seconds of smoothing
    lag right at the transition -- see the trim_uphill_overshoot saga in
    earlier iterations of this pipeline), take a fixed-width window
    centered on the phase's midpoint instead. This sidesteps the boundary
    entirely: it doesn't matter if the edges of the detected phase are
    imprecise by a few seconds, because the fit window never touches them.
    Requires min_duration > fit_window_sec (default 45s > 30s) so the
    window always fits inside a detected phase.
    """
    print(f"\n{'=' * 60}\nProcessing: {file_path}\n{'=' * 60}")

    t, o2 = load_o2_trace(file_path)

    dt = np.median(np.diff(t))
    window = max(int(round(window_seconds / dt)) | 1, polyorder + 2)
    o2_smooth = savgol_filter(o2, window, polyorder)
    rate = savgol_filter(o2, window, polyorder, deriv=1, delta=dt) * 60

    # Adaptive threshold: scales to this file's own noise level (raw minus
    # smoothed O2), not a fixed magnitude -- validated in earlier iterations
    # of this pipeline to generalize across files with very different
    # amplitude scales, where a fixed threshold either fragmented on noise
    # or swallowed real phases into 'flat'.
    threshold = 4 * (o2 - o2_smooth).std()
    phase = np.where(rate > threshold, "uphill", np.where(rate < -threshold, "downhill", "flat"))
    boundaries = np.concatenate([[0], np.where(phase[1:] != phase[:-1])[0] + 1, [len(t)]])

    rough_segments = []
    for i in range(len(boundaries) - 1):
        s_idx, e_idx = boundaries[i], boundaries[i + 1] - 1
        if (t[e_idx] - t[s_idx]) >= min_duration and phase[s_idx] in ["uphill", "downhill"]:
            rough_segments.append((s_idx, e_idx, phase[s_idx]))

    segments = []
    rates = []
    half_win = int(round((fit_window_sec / 2) / dt))

    for s_idx, e_idx, direction in rough_segments:
        mid_idx = s_idx + (e_idx - s_idx) // 2
        w_s = max(0, mid_idx - half_win)
        w_e = min(len(t) - 1, mid_idx + half_win)
        segments.append((w_s, w_e, direction))

        slope, _ = np.polyfit(t[w_s:w_e + 1], o2[w_s:w_e + 1], 1)
        rates.append({"dir": direction, "rate": slope * 60, "start": t[w_s] / 60, "end": t[w_e] / 60})

    r_df = pd.DataFrame(rates)
    if not r_df.empty:
        print("Rates per Segment (30s window):")
        for i, row in r_df.iterrows():
            print(f"  Segment {i + 1} ({row['dir']}) [{row['start']:.1f}-{row['end']:.1f} min]: "
                  f"{row['rate']:.4f} \u0394O2/min")

        export_df = r_df.copy()
        export_df.index = np.arange(1, len(export_df) + 1)
        export_df.index.name = "Segment"
        export_df = export_df.rename(columns={
            "dir": "Direction", "rate": "Rate_dO2_per_min",
            "start": "Start_min", "end": "End_min"
        })
        if save:
            csv_name = f"{Path(file_path).stem}_rates.csv"
            export_df.to_csv(csv_name)
            print(f"Saved {csv_name}")
    else:
        print("No uphill/downhill segments found "
              "(check threshold/min_duration if this is unexpected).")

    fig, (ax_o2, ax_rate, ax_seg) = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    t_min = t / 60

    ax_o2.plot(t_min, o2, color=COLORS["raw"], lw=0.6, alpha=0.7, label="Raw")
    ax_o2.plot(t_min, o2_smooth, color=COLORS["smooth"], lw=1.6, label="Smoothed")
    ax_o2.set_ylabel("Oxygen (raw instrument units)")
    ax_o2.set_title("Kinetic O2 Trace")
    ax_o2.legend(loc="best")
    ax_o2.grid(alpha=0.3)

    ax_rate.plot(t_min, rate, color=COLORS["rate"], lw=1.2, label="Calculated Rate")
    ax_rate.axhline(0, color="black", lw=0.5, ls="--")
    ax_rate.set_ylabel("Rate (\u0394O2/min)")
    ax_rate.set_title("Instantaneous Rate")
    ax_rate.grid(alpha=0.3)

    ax_seg.plot(t_min, o2, color=COLORS["raw"], lw=0.6, alpha=0.7)
    for s_idx, e_idx, direction in segments:
        ax_seg.plot(t_min[s_idx:e_idx + 1], o2_smooth[s_idx:e_idx + 1], color=COLORS[direction], lw=3.0)

    handles = [Line2D([0], [0], color=COLORS[d], lw=3.0, label=d) for d in ["uphill", "downhill"]]
    if handles:
        ax_seg.legend(handles=handles, loc="best")
    ax_seg.set_xlabel("Time (min)")
    ax_seg.set_ylabel("Oxygen (raw instrument units)")
    ax_seg.set_title("Photosynthesis and Respiration rates (30s fit windows)")
    ax_seg.grid(alpha=0.3)

    plt.tight_layout()
    out_name = f"{Path(file_path).stem}_analysis.svg"
    if save:
        plt.savefig(out_name, format="svg")
        plt.savefig(out_name.replace(".svg", ".png"), dpi=150)
        print(f"Saved {out_name} and {out_name.replace('.svg', '.png')}")
    if show:
        plt.show()
    plt.close(fig)

    return t, o2, o2_smooth, rate, Path(file_path).stem


def plot_overlay(datasets, output_name="combined_o2_kinetics_overlay", save=True, show=False):
    print(f"\n{'=' * 60}\nBuilding Combined Overlay Plot\n{'=' * 60}")
    fig, (ax_o2, ax_rate) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1]})

    for i, (t, o2, o2_smooth, rate, label) in enumerate(datasets):
        t_min = t / 60
        colour = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        ax_o2.plot(t_min, o2, color="#999999", lw=0.4, alpha=0.4)
        ax_o2.plot(t_min, o2_smooth, color=colour, lw=1.6, label=label)
        ax_rate.plot(t_min, rate, color=colour, lw=1.2, label=label)

    ax_o2.set_ylabel("Oxygen (raw instrument units)")
    ax_o2.set_title("O2 Kinetic Trace Overlay")
    ax_o2.legend(loc="best", fontsize=9)
    ax_o2.grid(alpha=0.3)

    ax_rate.axhline(0, color="black", lw=0.8)
    ax_rate.set_xlabel("Time (min)")
    ax_rate.set_ylabel("Rate (\u0394O2/min)")
    ax_rate.set_title("Instantaneous Rate Overlay")
    ax_rate.legend(loc="best", fontsize=9)
    ax_rate.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(f"{output_name}.svg", format="svg")
        plt.savefig(f"{output_name}.png", dpi=150)
        print(f"Saved {output_name}.svg and {output_name}.png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# EDIT THIS LIST IF RUNNING IN JUPYTER / IPYTHON
# (a notebook cell can't read command-line arguments -- sys.argv there holds
# the kernel's own startup flags like '-f', not your filenames)
# ---------------------------------------------------------------------------
INPUT_FILES = [
    "your_file_1.csv",
    "your_file_2.csv",
]


if __name__ == "__main__":
    running_in_notebook = "ipykernel" in sys.argv[0] or "ipykernel" in " ".join(sys.argv)

    if running_in_notebook:
        input_files = INPUT_FILES
    elif len(sys.argv) < 2:
        print("Usage: python o2_kinetics_pipeline.py file1.csv [file2.csv file3.csv ...]")
        sys.exit(1)
    else:
        input_files = sys.argv[1:]

    datasets = []
    for f in input_files:
        try:
            datasets.append(process_and_plot(f))
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if datasets:
        plot_overlay(datasets)
