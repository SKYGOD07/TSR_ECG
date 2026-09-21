import os
import sys
import ast
import copy
import argparse
import numpy as np
import pandas as pd
import wfdb
import heartpy as hp
from tqdm import tqdm

def load_raw_data(df, sampling_rate, path):
    col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df[col], desc='Reading WFDB signals')]
    return np.array([signal for signal, meta in data])

def preprocess_ptbxl(path, sampling_rate=500):
    print(f"Loading database CSVs from {path}...")
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    X = load_raw_data(Y, sampling_rate, path)

    agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == True]

    def aggregate_diagnostic(y_dic):
        tmp = [agg_df.loc[k].diagnostic_class for k in y_dic.keys() if k in agg_df.index]
        return list(set(tmp))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    test_fold = 10
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    train_data, count = [], 0
    for item in y_train:
        try:
            if item[0] == 'NORM':
                train_data.append(X_train[count])
            count += 1
        except Exception:
            count += 1
    train_data = np.asarray(train_data)

    # test_class keeps the raw PTB-XL diagnostic superclass (NORM/MI/STTC/CD/HYP)
    # alongside the binary label. It is additive: label.npy keeps exactly the same
    # 0/1 meaning it always had. Without it, anomaly-specific timestep analysis
    # (Stage 4) would have nothing to condition on -- and classes must never be
    # invented to fill that gap.
    test_label, test_data, test_class, count = [], [], [], 0
    for item in y_test:
        try:
            test_label.append(0 if item[0] == 'NORM' else 1)
            test_class.append(str(item[0]))
            test_data.append(X_test[count])
            count += 1
        except Exception:
            count += 1
    test_label = np.asarray(test_label)
    test_data = np.asarray(test_data)
    test_class = np.asarray(test_class, dtype=object)

    print(f"train_data: {train_data.shape}, test_data: {test_data.shape}, "
          f"test_label: {test_label.shape} (abnormal count: {test_label.sum()})")
    return train_data, test_data, test_label, test_class

def normalize(X_ori):
    X = copy.deepcopy(X_ori)
    for n in range(X.shape[0]):
        for lead in range(12):
            seq = X[n][:, lead]
            seq_min, seq_max = seq.min(), seq.max()
            if seq_max > seq_min:
                X[n][:, lead] = 2 * (seq - seq_min) / (seq_max - seq_min) - 1
            else:
                X[n][:, lead] = 0
    return X

def hp_preprocess(X):
    out = []
    for i in tqdm(range(X.shape[0]), desc='Filtering signals'):
        leads = []
        for lead in range(12):
            ecg = X[i][:, lead]
            f1 = hp.filter_signal(ecg, sample_rate=500, filtertype='highpass', cutoff=1)
            f2 = hp.filter_signal(f1, sample_rate=500, cutoff=35, filtertype='notch')
            f3 = hp.filter_signal(f2, sample_rate=500, filtertype='lowpass', cutoff=25)
            leads.append(f3)
        out.append(np.array(leads).T)
    return np.array(out)

def denoise_train(train_data, out_dir):
    denoised = normalize(hp_preprocess(train_data))
    kept = []
    for i in range(denoised.shape[0]):
        try:
            hp.process(denoised[i, :, 1], 500.0)
        except Exception:
            continue
        kept.append(denoised[i])
    kept = np.array(kept)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'train.npy')
    np.save(out_file, kept)
    print(f"Saved {out_file} ({kept.shape})")

def denoise_test(test_data, test_label, out_dir, test_class=None):
    denoised = hp_preprocess(test_data)
    data_kept, label_kept, class_kept = [], [], []
    for i in range(denoised.shape[0]):
        try:
            hp.process(denoised[i, :, 1], 500.0)
        except Exception:
            continue
        data_kept.append(denoised[i])
        label_kept.append(test_label[i])
        if test_class is not None:
            class_kept.append(test_class[i])
    data_kept = np.array(data_kept)
    label_kept = np.array(label_kept)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'test.npy'), data_kept)
    np.save(os.path.join(out_dir, 'label.npy'), label_kept)
    print(f"Saved {os.path.join(out_dir, 'test.npy')} ({data_kept.shape}) and label.npy ({label_kept.shape})")
    if test_class is not None:
        class_kept = np.array(class_kept, dtype=object)
        # Same row order and the same kept/dropped rows as test.npy and label.npy.
        np.save(os.path.join(out_dir, 'test_class.npy'), class_kept, allow_pickle=True)
        classes, counts = np.unique(class_kept.astype(str), return_counts=True)
        print(f"Saved {os.path.join(out_dir, 'test_class.npy')} ({class_kept.shape}) "
              f"-> {dict(zip(classes.tolist(), counts.tolist()))}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw PTB-XL dataset into TSRNet npy format")
    parser.add_argument("--raw_path", type=str, required=True, help="Path to raw PTBXL folder containing ptbxl_database.csv")
    parser.add_argument("--out_dir", type=str, default="data/", help="Output directory for train.npy, test.npy, label.npy")
    parser.add_argument("--sampling_rate", type=int, default=500, help="Sampling rate (500 or 100)")
    args = parser.parse_args()

    train_data, test_data, test_label, test_class = preprocess_ptbxl(args.raw_path, args.sampling_rate)
    print("Denoising + normalizing train set...")
    denoise_train(train_data, args.out_dir)
    print("Denoising test set...")
    denoise_test(test_data, test_label, args.out_dir, test_class=test_class)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
