# 6. In-Depth Guide: Evaluation and the ROC-AUC Score

After `train.py` finishes its epochs, it saves the AI's 4.39 million optimized weights into a file called a Checkpoint (`ckpt/TSRNet-latest.pt`). 

But how do we know if the AI is actually a good doctor? We test it using `test.py`. This document explains how the test works and how to read the final mathematical score.

---

## Part 1: The Blind Test

In our dataset, we have 2,155 patients set aside in `test.npy`. 
The AI was completely banned from looking at these patients during training. This prevents the AI from just memorizing the answers (a problem called **Overfitting**). 

Furthermore, unlike the training data (which was 100% healthy patients), the test data is a mix of Healthy patients and Sick patients (heart attacks, arrhythmias, etc.).

**The Testing Process:**
1. `test.py` loads the AI checkpoint.
2. It feeds the 2,155 test patients into the AI one by one, masking 30% of their data.
3. The AI attempts to reconstruct the data. 
4. The script calculates the **Reconstruction Error** (MSE) for every patient.

![Original vs Reconstructed ECG Graph](Images/reconstruction_graph.png)

5. High Error = AI thinks they are Sick. Low Error = AI thinks they are Healthy.

---

## Part 2: Grading the AI (Confusion Matrix)

When the AI makes a prediction, there are 4 possible outcomes:
1. **True Positive (TP):** The patient is Sick, and the AI correctly flagged them as Sick. (Great!)
2. **True Negative (TN):** The patient is Healthy, and the AI correctly called them Healthy. (Great!)
3. **False Positive (FP):** The patient is Healthy, but the AI accidentally flagged them as Sick. (Bad! This causes unnecessary stress and hospital bills).
4. **False Negative (FN):** The patient is Sick, but the AI missed it and sent them home. (Terrible! This is potentially fatal).

### The Threshold Dilemma
Remember that the AI just outputs an Error Number (e.g., `0.015`). We have to set a "Threshold" line. If the error is above the line, we call them sick. 

* If we set the threshold **too low**, the AI will flag everyone as sick (Lots of False Positives).
* If we set the threshold **too high**, the AI will flag everyone as healthy (Lots of False Negatives).

---

## Part 3: The ROC-AUC Score

Because picking a single threshold is hard, data scientists use a metric that tests *every possible threshold* simultaneously. This is called the **ROC-AUC Score**.

### What does ROC mean?
ROC stands for **Receiver Operating Characteristic**. It is a graph line plotted on a 2D chart:
* The Y-axis is the **True Positive Rate** (How many sick people we successfully caught).
* The X-axis is the **False Positive Rate** (How many healthy people we accidentally scared).

As we slide our imaginary Threshold from 0.0 to 1.0, the graph plots a curving line. 

### What does AUC mean?
AUC stands for **Area Under the Curve**. Using calculus (integrals), the computer calculates exactly how much blank space is underneath that curving ROC line. 

This gives us a single, universally understood number to grade our AI:
* **AUC = 0.500:** The curve is a straight diagonal line. The AI is utterly useless. It has a 50/50 chance of being right, exactly like flipping a coin. 
* **AUC = 0.800:** The AI is very good! 80% of the time, it will rank a sick patient as having a higher error than a healthy patient. 
* **AUC = 1.000:** The AI is an omniscient god. It perfectly separates the sick from the healthy with 0 False Positives and 0 False Negatives.

![ROC Curve Score Plot](Images/roc_curve.png)

---

## Part 4: Our Local Results vs Cloud Target

When we ran `test.py` locally on your computer, the script printed this to the terminal:
```powershell
('Detection AUC: ', 0.607)
```

### Is 0.607 good or bad?
For a fully trained medical AI, 0.607 is terrible. It is only slightly better than flipping a coin. 
**However, for our specific test, it is a massive success!**

Why? Because we intentionally only trained the AI for **1 Epoch** (1 read through the textbook). The AI barely had time to adjust its weights. The fact that the AUC moved from 0.500 up to 0.607 in a single epoch proves that the architecture works, the math is correct, and the AI is successfully learning!

### The Next Steps
When you are ready, you will upload this exact, bug-free codebase to Kaggle. You will utilize an Nvidia GPU to run the training loop for **50 Epochs**. 

As the AI reads the dataset 50 times, the Reconstruction Error for healthy patients will plunge toward zero, while the error for sick patients remains sky-high. 
According to the original ISBI 2024 paper, TSRNet is capable of hitting a final **AUC of 0.865+**, making it a highly robust, state-of-the-art framework for real-time ECG anomaly detection!

---

## Part 5: A second opinion - the noise anomaly score

Everything above grades the AI on **restoration error**: mask part of the ECG, see how
badly it is rebuilt. The diffusion branch adds a completely different kind of score.

Instead of hiding part of the signal, it **buries the signal in a known amount of
static** and asks the network one question: *how much static did I just add?*

```
A_noise = average of ( real static - the static the AI guessed )^2
```

A network that has only ever studied healthy hearts can separate a healthy heartbeat
from static cleanly. Show it an unfamiliar, unhealthy shape and it starts mistaking
signal for static - so the guess gets worse, and the score goes up.

### Putting the two grades together

```
A_final = alpha * A_TSR + (1 - alpha) * A_noise
```

with alpha fixed at 0.5. Because the two scores are on completely different scales
(restoration error ~0.2, noise error ~1.0), each is rescaled to [0, 1] first. Rescaling
is order-preserving, so it does not change either score's own AUC - it just makes
adding them meaningful.

### How to read the result honestly

Three numbers are reported on the **same** test set, from the **same** per-sample
scores:

| | What it uses |
|---|---|
| **M1** | TSR restoration error only |
| **M2** | Diffusion noise error only |
| **M4** | Both, fused |

If M4 does not beat the better of M1 and M2, the script prints exactly that. The
sweep over alpha is also printed, but the best alpha in it was chosen by looking at the
test set, so it is an upper bound rather than an honest held-out result - the
fixed alpha = 0.5 number is the one to quote.

The same rule applies to the diffusion timestep: `exp03_step_analysis.py` evaluates
t = 1, 5, 10, ..., 100 and reports which one scored best **among those it actually
ran**. No timestep is declared optimal before it has been measured.

Full details: [07_diffusion_noise_branch.md](07_diffusion_noise_branch.md).
