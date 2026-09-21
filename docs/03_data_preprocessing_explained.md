# 3. In-Depth Guide: Data Preprocessing (Cleaning the Data)

Before we can train the AI, we have to prepare the raw hospital data. If you feed garbage into an AI, you get garbage out. This is why data preprocessing is often the most critical part of machine learning.

This document breaks down exactly what `preprocess.py` does to the massive 60,000-number matrices we get from the PTB-XL dataset.

---

## Part 1: The Physics of ECG Noise

When a hospital records an ECG, the sensors are incredibly sensitive. They are designed to pick up micro-volts of electricity from your heart. Unfortunately, they also pick up everything else.

Here are the three main enemies of a clean ECG signal:

### 1. Powerline Interference (50 Hz or 60 Hz Noise)
Hospitals are filled with alternating current (AC) electricity powering lights, beds, and monitors. In the US, this electricity pulses at 60 Hz (60 times a second). In Europe/Asia, it's 50 Hz.
This creates a constant, fast "fuzz" or "buzz" over the entire ECG signal.

### 2. Baseline Wander (< 1 Hz)
As the patient breathes in, their lungs expand, pushing the heart and chest up. When they exhale, the chest goes down. 
This causes the entire ECG line to slowly drift up and down the page like a rolling ocean wave. This is called Baseline Wander.

### 3. EMG Noise (Electromyography Noise)
If the patient shivers, talks, or moves their arms, their skeletal muscles generate electricity. This creates random, high-frequency, jagged spikes on the graph.

---

## Part 2: Digital Signal Filtering (How we clean it)

To clean the data, `preprocess.py` uses digital **Butterworth Filters** from the Python `scipy.signal` library. 
A filter is just a mathematical formula that deletes certain frequencies (speeds) of data while letting others pass through safely.

### The Code Breakdown
If you look inside the preprocessing code, you'll see steps that apply these filters to our `(14241, 12, 4800)` array:

1. **High-Pass Filter (Cutoff: 1 Hz):**
   * **What it does:** It acts like a bouncer that only lets "fast" signals pass through. It deletes anything moving slower than 1 cycle per second.
   * **Why:** This perfectly eliminates the slow, rolling ocean wave caused by patient breathing (Baseline Wander).
   
2. **Low-Pass Filter (Cutoff: 25-35 Hz):**
   * **What it does:** It only lets "slow" signals pass through, deleting anything vibrating extremely fast.
   * **Why:** The human heartbeat rarely has meaningful electrical information above 25 Hz. By cutting off everything above that, we instantly delete the 50/60 Hz Powerline "fuzz" and the jagged EMG muscle spikes!

After filtering, the squiggly line is perfectly flat, smooth, and only shows the pure P-QRS-T heartbeats.

---

## Part 3: Normalization (Making the data fair)

Even after cleaning, we have another problem. 
Patient A might have a strong heart that generates voltages between `-5.0` and `+5.0`. 
Patient B might have a weak heart that generates voltages between `-0.5` and `+0.5`.

If we feed this to the AI, the AI will prioritize the larger numbers because math models naturally weight big numbers heavier. We need to **Normalize** the data so every patient is on the exact same scale.

### Min-Max Scaling
In our project, we use a technique called Min-Max Scaling (often scaled to range -1 to 1, or 0 to 1). 
The math formula looks like this:

```math
Scaled\_Value = \frac{Current\_Voltage - Minimum\_Voltage}{Maximum\_Voltage - Minimum\_Voltage}
```

* For Patient A, their +5.0 becomes `1.0`, and their -5.0 becomes `0.0` (or `-1.0`).
* For Patient B, their +0.5 becomes `1.0`, and their -0.5 becomes `0.0`.

Now, both patients' heartbeats are perfectly stretched to the exact same height on the graph! The AI can now focus purely on the *shape* of the heartbeat, rather than how big the electrical current was.

---

## Part 4: Saving the Final Output

After `preprocess.py` finishes reading the `ptbxl_database.csv`, filtering the noise, and normalizing the voltages, it saves the data into Python NumPy arrays.

* **`data/train.npy`**: The massive 3D array of our 14,241 healthy patients `(14241, 12, 4800)`.
* **`data/test.npy`**: The array of our 2,155 test patients.
* **`data/label.npy`**: A list of the correct answers for the test patients (0 for Healthy, 1 for Sick) so we can grade the AI's final exam.

Because these are saved as binary `.npy` files instead of text, PyTorch can load them into memory in fractions of a second when training begins!

---

## Part 5: Two things that were added later

### `data/test_class.npy`

Alongside `label.npy` (0 = healthy, 1 = sick), `preprocess.py` now also saves the raw
PTB-XL **diagnostic superclass** for each test patient:

| Code | Meaning |
|---|---|
| `NORM` | normal |
| `MI` | myocardial infarction |
| `STTC` | ST/T-wave changes |
| `CD` | conduction disturbance |
| `HYP` | hypertrophy |

This is purely additive - `train.npy`, `test.npy` and `label.npy` are exactly what they
always were. It exists so the diffusion branch can ask whether *different kinds* of
abnormality are easiest to spot at different noise levels. Without it, that question
cannot be answered honestly, and the right response is to say so rather than invent
class labels.

### A wrinkle worth knowing: train and test are not scaled the same way

Look carefully at `preprocess.py`:

* `denoise_train` runs `normalize(hp_preprocess(...))` - filtering **and** min-max scaling.
* `denoise_test` runs `hp_preprocess(...)` alone - filtering **only**.

So `train.npy` lives neatly in [-1, 1] (std ~0.56), while `test.npy` keeps its original
millivolt amplitudes (std ~0.20, with outliers past +/-5). This comes from the original
repository and TSR-Net has always been evaluated this way.

It matters much more for the diffusion branch, because diffusion mixes the signal with
fixed-size noise - so the *ratio* between them depends directly on how big the signal
is. The diffusion code therefore re-applies the same min-max scaling to test windows
**at evaluation time**, leaving the files on disk untouched. See
[07_diffusion_noise_branch.md](07_diffusion_noise_branch.md) section 11.

### One more time: filtering is not diffusion noise

Everything on this page is about **removing** unwanted noise before the AI ever sees
the signal. The diffusion branch **adds** noise on purpose, afterwards, and asks the
network to identify it. They are opposite operations at opposite ends of the pipeline
and should never be confused. See
[07_diffusion_noise_branch.md](07_diffusion_noise_branch.md) section 2.
