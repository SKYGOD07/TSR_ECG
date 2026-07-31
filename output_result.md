# TSRNet Local Benchmark Execution Output

## Training Results (`train.py`)
- **Total Trainable Parameters:** 2.17 M
- **Epoch 0:** Total_Loss: 1.218870
- **Epoch 1:** Total_Loss: 0.997968
- **Epoch 2:** Total_Loss: 0.951159

## Inference & Benchmarking Results (`test.py`)
- **Dataset Size:** 4 samples (dummy data)
- **Time/Cost:** < 10 seconds execution time on CPU
- **Benchmark Score (ROC AUC):** 1.0

### Benchmark Score Summary
The benchmark score represents the Area Under the Receiver Operating Characteristic Curve (ROC AUC). A score of **1.0** indicates that the model perfectly distinguished the anomaly sample from the normal samples. While this score is perfect, it was achieved on a small synthetic dummy dataset for validation purposes. 
