#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #f3e8ff; border: 1px solid #d8b4fe; color: #6b21a8; padding: 15px; border-radius: 4px;">
# <b> Oxygen data processing </b> <br> 
# First code is the complete one, whilst the second is just for the overlay
# </div>

# In[1]:


import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

# <div style="background-color: #f3e8ff; border: 1px solid #d8b4fe; color: #6b21a8; padding: 15px; border-radius: 4px;">
# <b> Overlay code </b> <br> 
# Only the overlay chunk
# </div>

# In[4]:


def find_header_row(file_path, max_scan=60):
    with open(file_path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_scan:
                break
            # Explicitly match "oxygen 1" to skip the uppercase general header
            if line.strip().lower().startswith("time,oxygen 1"):
                return i
    raise ValueError(f"Could not find a 'Time,Oxygen 1' header row in {file_path}.")


def load_o2_trace(file_path):
    header_row = find_header_row(file_path)
    df = pd.read_csv(file_path, skiprows=header_row, index_col=False)
    df.columns = df.columns.str.strip()

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Oxygen 1"] = pd.to_numeric(df["Oxygen 1"], errors="coerce")
    df = df.dropna(subset=["Time", "Oxygen 1"]).sort_values("Time").reset_index(drop=True)

    return df["Time"].values, df["Oxygen 1"].values


def compute_rate(t, o2, window_seconds=31, polyorder=3):
    dt = np.median(np.diff(t))  
    window = int(round(window_seconds / dt))
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2)

    o2_smooth = savgol_filter(o2, window_length=window, polyorder=polyorder)
    rate_per_sec = savgol_filter(o2, window_length=window, polyorder=polyorder,
                                  deriv=1, delta=dt)
    rate_per_min = rate_per_sec * 60

    return o2_smooth, rate_per_min


def plot_kinetic_and_rate(datasets, output_name):
    fig, (ax_o2, ax_rate) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # Color palettes for multiple files
    colors_o2 = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]
    colors_rate = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]

    for i, (t, o2, o2_smooth, rate_per_min, label) in enumerate(datasets):
        t_min = t / 60
        c_o2 = colors_o2[i % len(colors_o2)]
        c_rate = colors_rate[i % len(colors_rate)]

        # Top plot: O2 trace
        ax_o2.plot(t_min, o2, color="#999999", linewidth=0.4, alpha=0.4) 
        ax_o2.plot(t_min, o2_smooth, color=c_o2, linewidth=1.6, label=label)

        # Bottom plot: Rate
        ax_rate.plot(t_min, rate_per_min, color=c_rate, linewidth=1.2, label=label)

    ax_o2.set_ylabel("Oxygen Concentration (nmol)") 
    ax_o2.set_title("O2 kinetic trace")
    ax_o2.legend(loc="best", fontsize=9)
    ax_o2.grid(alpha=0.3)

    ax_rate.axhline(0, color="black", linewidth=0.8)
    ax_rate.set_xlabel("Time (min)")
    ax_rate.set_ylabel("Rate (nmol / ml / min)")
    ax_rate.set_title("Instantaneous rate")
    ax_rate.legend(loc="best", fontsize=9)
    ax_rate.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_name}.svg", format="svg")
    print(f"Saved {output_name}.svg")


# In[7]:


if __name__ == "__main__":
    # 1. List your files directly here
    input_files = [
        "Dummy_CONTROL.csv",
        "Dummy_CONDITION.csv"
    ]


    output_name = "O2_kinetic_comparison"
    datasets = []

    # 2. Loop through the list to load and compute rates
    for file_path in input_files:
        t, o2 = load_o2_trace(file_path)
        o2_smooth, rate_per_min = compute_rate(t, o2)

        # Strip '.csv' from the filename for a cleaner legend label
        label = file_path.replace(".csv", "")
        datasets.append((t, o2, o2_smooth, rate_per_min, label))

    # 3. Plot the combined dataset list
    plot_kinetic_and_rate(datasets, output_name)

