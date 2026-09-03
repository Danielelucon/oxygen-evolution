Clark Pipeline: Oxygen Evolution Analysis

A Python-based workflow for processing, smoothing, and analyzing dissolved oxygen kinetic data from Clark-type electrodes.

This repository provides tools to automatically filter raw electrode noise, detect active phases of photosynthesis and respiration, calculate precise physiological rates, and generate publication-ready figures.

🧰 Repository Structure

This repository is split into two primary tools depending on your analytical needs:

Full Analysis Pipeline (o2_analysis.py)

Designed for rigorous rate extraction.

Processes raw kinetic traces and applies Savitzky-Golay filtering to calculate continuous instantaneous rates.

Automatically detects rough metabolic phases based on standard deviation thresholds.

Isolates the geometric midpoint of these phases and applies a strict 30-second linear regression to calculate highly accurate $\Delta O_2/min$ rates.

Exports a detailed .csv table of all calculated rates and a 3-panel .svg summary plot.

Overlay Visualization (o2_overlay.py)

Designed for rapid visual comparison across multiple biological replicates or conditions.

Bypasses the strict regression math to purely focus on smoothing and plotting.

Generates a clean, 2-panel overlay (Kinetic Trace + Instantaneous Rate) for any number of input files using a colourblind-safe Okabe-Ito palette.

📊 How It Works (The Math)

Clark electrode data is notoriously noisy, making automated linear regressions difficult. This pipeline solves that using the following approach:

Smoothing & Derivation: The raw oxygen concentration (nmol) is passed through a Savitzky-Golay filter (default: polyorder 3). The pipeline simultaneously calculates the first derivative to determine the instantaneous rate of change per second, scaled to per minute.

Phase Detection: A dynamic threshold (4× the standard deviation of the noise) is used to locate active "uphill" (photosynthetic) and "downhill" (respiratory) phases that last for a minimum duration.

Midpoint Regression: To avoid the "lag" often seen at the beginning and end of light/dark transitions, the algorithm finds the exact geometric midpoint of the active phase. It then creates a strict 30-second window around this midpoint and calculates a linear regression (slope) to determine the final physiological rate.

🚀 Usage

Data Requirements

Input data should be standard .csv exports from your electrode software. The script dynamically scans the file for the start of the data block, looking for the column headers Time and Oxygen 1.

Running the Analysis

Simply add your .csv files to the input_files list at the bottom of either script:

input_files = [
    "Dummy_CONTROL.csv",
    "Dummy_CONDITION.csv"
]
