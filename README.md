# Oxygen Evolution Analysis

This repository provides a downstream analysis pipeline for dissolved oxygen kinetic data obtained from Clark-type electrodes.

The pipeline parses raw electrode outputs, applies Savitzky-Golay filtering, automatically detects active metabolic phases, calculates physiological rates via linear regression, and generates overlay visualisations.

Please take note that the "header" function should be modified to suit your instrument output. This one will work for the old Hansatech Clark electrode.

## Repository Contents

* Universal_Oxygen_Rate_Analysis.ipynb: The primary Jupyter Notebook combining both analysis and overlay functions.
* o2_analysis.py: A standalone Python script for full rate extraction and individual multi-panel plotting.
* o2_overlay.py: A standalone Python script for rapid visual overlay of multiple traces.
* Dummy_CONTROL.csv & Dummy_CONDITION.csv: Anonymised datasets for local testing.

## Mandatory Data Structure

Input data must be standard .csv exports from your electrode software. The script dynamically scans the file to locate the start of the data block, but requires the following exact column headers (case-insensitive, to be modified based on the instrument output):

* Time
* Oxygen 1

## Local Execution

### 1. Install Dependencies

Install the required libraries via your terminal:
pip install numpy pandas scipy matplotlib

### 2. Run the Code

**Option A: Using standard Python**
Open o2_analysis.py or o2_overlay.py in your preferred IDE, update the input_files list at the very bottom of the script with your file paths, and execute.

**Option B: Using Jupyter Notebook**
Launch the notebook environment from your terminal:
jupyter notebook
Open Universal_Oxygen_Rate_Analysis.ipynb, modify the input_files list in the execution cell, and run the notebook.
