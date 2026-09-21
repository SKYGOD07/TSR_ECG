# TSRNet Beginner Documentation

Welcome! This folder contains a step-by-step, plain-English breakdown of the entire TSRNet project. If you are new to deep learning, medical data, or programming, these guides will explain exactly what this project does and how it works.

## Table of Contents

1. **[What is an ECG and the PTB-XL Dataset?](01_what_is_an_ecg.md)**  
   *Learn about heartbeats, 12-lead sensors, and where our data comes from.*

2. **[What is AI Anomaly Detection?](02_what_is_ai_anomaly_detection.md)**  
   *Understand how Artificial Intelligence can spot a sick heart.*

3. **[Data Preprocessing (Cleaning the Data)](03_data_preprocessing_explained.md)**  
   *Why hospital data is messy and how we clean it using filters.*

4. **[The TSRNet Architecture (How the AI works)](04_the_tsrnet_architecture.md)**  
   *The "Two Brains" of our AI and how it solves puzzles to learn.*

5. **[Training, Epochs, and the Range Error](05_training_and_epochs.md)**  
   *How the AI studies the data, and how we fixed a bug in the code.*

6. **[Evaluation and the ROC-AUC Score](06_evaluation_and_scores.md)**  
   *How we grade the AI's final exam and what the scores mean.*

7. **[The Diffusion / Noise Branch](07_diffusion_noise_branch.md)**  
   *The new research extension: adding controlled noise on purpose, asking the network
   to name it, and using the mismatch as a second anomaly score. Covers Stages 0-5,
   the exact maths, how to run it, and the known limitations.*

---

## Reports

- **[LOCAL_RUN_REPORT.md](LOCAL_RUN_REPORT.md)** - local execution metrics
- **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** - analysis notes
- **[output_result.md](output_result.md)** - benchmark output
- **[testing.md](testing.md)** - testing notes
