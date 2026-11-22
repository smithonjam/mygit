# mygit
# SSVEP EEG Simulator
## Overview
This is a Python simulation tool for SSVEP (Steady-State Visual Evoked Potential) EEG signals. It can generate simulated SSVEP brainwave data with realistic artifacts, perform bandpass filtering and FFT spectrum analysis, save raw signals into CSV files, export spectrum plots, and maintain experiment logs. It is suitable for algorithm verification, offline decoding testing, and demo of SSVEP-BCI research.

## Features
- Generate simulated SSVEP signal with target stimulation frequency
- Simulate common EEG components: alpha rhythm, Gaussian background noise, slow baseline drift, eye-blink artifacts
- 4th-order Butterworth bandpass filter (customizable frequency range)
- FFT transformation and amplitude spectrum calculation
- Batch trials for multiple stimulation frequencies
- Automatic export: time-series CSV data, spectrum figure (PNG)
- JSON experiment log to record all trial metadata
- Summary viewer for historical experiment records

## Environment & Dependencies
Python >= 3.10 (Tested on Python 3.12)
Install required packages:
```bash
pip install numpy scipy matplotlib pandas
```

## Project Structure
```
SSVEP_Simulator/
├── ssvep_sim.py          # Main source code
└── ssvep_records/        # Auto-generated output folder
    ├── experiment_log.json
    ├── ssvep_xxhz_xxx.csv
    └── plot_xxhz_xxx.png
```

## Quick Start
1. Install dependencies following the command above
2. Run the script directly
```bash
python ssvep_sim.py
```
3. The default batch task will run trials for stimulation frequencies: `[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`
4. All output files and logs will be saved under `./ssvep_records/`

## Core Class: SSVEP_Simulator
### Constructor Parameters
| Parameter | Type | Description |
| ---- | ---- | ---- |
| sample_rate | int | Sampling rate of EEG signal (Hz) |
| duration | float | Time length of a single trial (seconds) |

### Main Methods
- `generate_ssvep_signal(target_freq)`: Generate composite SSVEP raw signal
- `filter_bandpass(sig, low, high)`: Bandpass filter for EEG
- `compute_fft_spectrum(signal_data)`: Calculate FFT amplitude spectrum
- `run_single_trial(target_frequency)`: Run one independent trial
- `run_batch_experiment(freq_list)`: Run multiple stimulation frequencies in sequence
- `save_signal_csv(sig, freq, note)`: Export time series to CSV
- `plot_signal_and_spectrum(...)`: Plot raw signal, filtered signal and spectrum
- `load_log_and_summary()`: Load and print experiment log table

## Notes
- The simulation is for algorithm validation only, not real clinical EEG data.
- The eye-blink artifact, noise amplitude and baseline parameters can be modified inside `__init__`.
- If you need SSVEP decoding module (CCA / TRCA), extended code can be added.
- Matplotlib will pop up figure windows after each trial by default.

## License
MIT License
You can freely use, modify and distribute this code for academic research.

## Author
For SSVEP-BCI research.
