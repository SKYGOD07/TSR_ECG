import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

nb_path = 'TSRNet_Kaggle_Merge_Demo.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# First rename the existing sections appropriately to match STAGE 0
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        if 'Step 2 — Train TSRNet' in cell.source:
            cell.source = cell.source.replace('Step 2 — Train TSRNet', 'STAGE 0 — Existing TSR baseline: Train TSRNet')
        elif 'Step 3 — Get the ECG anomaly scores (AUC)' in cell.source:
            cell.source = cell.source.replace('Step 3 — Get the ECG anomaly scores (AUC)', 'STAGE 0 — Existing TSR baseline: Get Anomaly Scores')

cells_to_add = [
    new_markdown_cell("## STAGE 1 — Forward diffusion & STAGE 1 TEST — Noise visualization\nRun the smoke test to visualize noise addition on different timesteps."),
    new_code_cell("!python experiments/exp01_forward_noise.py\n\nfrom IPython.display import Image, display\ndisplay(Image(filename='Images/forward_diffusion_smoke_test.png'))"),
    
    new_markdown_cell("## STAGE 2 — Diffusion noise predictor\nTest the architecture of the diffusion noise predictor model on a small batch."),
    new_code_cell("!python experiments/exp02_noise_prediction.py"),
    
    new_markdown_cell("## STAGE 2 TRAIN — Normal-only training\nTrain the diffusion noise predictor exclusively on normal ECGs. We use 10 epochs for demonstration."),
    new_code_cell("!python training/train_diffusion.py --epochs 10 --batch_size 32"),
    
    new_markdown_cell("## STAGE 3 — Noise anomaly score & STAGE 4 — Timestep analysis\nEvaluate noise scores across multiple diffusion timesteps and compare normal vs abnormal."),
    new_code_cell("!python experiments/exp03_step_analysis.py\n\ndisplay(Image(filename='Images/step_analysis_scores.png'))\ndisplay(Image(filename='Images/step_analysis_auc.png'))"),
    
    new_markdown_cell("## STAGE 5 — TSR + Noise fusion\nEstablish individual baselines and then fuse the TSR anomaly score and the Noise anomaly score."),
    new_code_cell("!python experiments/exp04_tsr_vs_noise.py\n!python experiments/exp05_fusion.py")
]

nb.cells.extend(cells_to_add)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook updated successfully.")
