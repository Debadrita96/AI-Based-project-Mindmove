
# MindMove: Real-Time EMG-Based Hand Gesture Classification

Real-time decoding of surface Electromyography (sEMG) signals into hand movement intent using Support Vector Machines (SVM).

Developed as part of the "Mind Move" seminar at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

---

##  Overview

This project focuses on decoding neural signals (sEMG) to classify hand gestures using classical machine learning techniques. 

We implement:

- Signal preprocessing (bandpass + notch filtering)
- Time-domain feature extraction
- Frequency-domain feature extraction
- Wavelet transform features
- SVM classifier with RBF kernel
- Hyperparameter tuning using GridSearchCV
- Real-time EMG streaming simulation via TCP socket

The model achieves ~83–85% accuracy on Day 1 EMG dataset.

---

## Motivation

Surface EMG (sEMG) signals can be used for:

- Prosthetic limb control
- Human-computer interaction
- Gesture recognition
- Neuromuscular diagnostics
- Rehabilitation robotics

This project demonstrates how classical ML pipelines can robustly decode movement intent from high-dimensional EMG signals.

---

## Dataset Description

- 720 signal segments
- 320 channels
- 64 samples per segment
- Sampling frequency: 2048 Hz
- 4 gesture classes:
  - `fist_slow`
  - `index_slow`
  - `rest_slow`
  - `twoFPinch_slow`

Dataset stored as `.pkl` file (not included in repo due to size constraints).

---

##  Methodology

### 1️⃣ Signal Filtering

- Chebyshev Type-II Bandpass Filter (20–500 Hz)
- Notch filtering (50 Hz powerline interference)

---

### 2️⃣ Feature Extraction (35 Features per Movement)

#### Time-Domain Features
- Mean Absolute Value (MAV)
- RMS
- Variance
- Zero Crossing Rate
- Waveform Length
- Slope Sign Changes

#### Frequency-Domain Features
- Power Spectrum
- Median Frequency
- Mean Frequency
- Frequency Ratio

#### Advanced Features
- 16 AR coefficients (real + imaginary parts)
- 10 Wavelet coefficients

---

### 3️⃣ Model

- Support Vector Machine (SVC)
- RBF Kernel
- Hyperparameter tuning using GridSearchCV
- Parameters tuned:
  - C
  - gamma

---

##  Performance

### Day 1 EMG Dataset

| Metric     | Score |
|------------|--------|
| Accuracy   | ~0.84 |
| Precision  | ~0.84 |
| Recall     | ~0.84 |
| F1-Score   | ~0.83 |

Model generalizes moderately to Day 2 dataset.

---

## Confusion Matrix & Evaluation

- Normalized confusion matrix
- Precision / Recall / F1 per class
- Feature importance using permutation importance
- Time-series prediction visualization

---

##  Real-Time EMG Simulation

Includes TCP-based EMG streaming simulator:

`simulator.py`

Features:

- Real-time frame streaming
- Configurable sampling rate
- TCP socket server
- EMG chunk streaming

This enables testing gesture classification in near real-time setups.

---

## 🗂 Repository Structure
SVM_TESTING.py # Main training + evaluation script
simulator.py # Real-time EMG streaming server
notebooks/ # Data exploration notebooks
README.md
.gitignore

---

## Tech Stack

- Python
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- Socket Programming





