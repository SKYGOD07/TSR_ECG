import numpy as np
import os
import heartpy as hp
import copy
from scipy.signal import stft
import pywt
import torch
import random
import torchvision.transforms as transforms
from utils import normalize, beat_normalize
from torch.utils.data import Dataset
 
class TrainSet(Dataset):
    def __init__(self, folder, fs=500, nperseg=125):
        self.train_data = np.load(os.path.join(folder, 'train.npy'))
        self.fs = fs
        self.nperseg = nperseg

    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return peak_list
    
    def __len__(self):
        return self.train_data.shape[0]
 
    def __getitem__(self, index):
        time_instance = self.train_data[index]
        time_instance = time_instance[100:4900,:] #(4800, 12)
       
        # Short Time Fast Fourier Transform
        # 500 is the sample rate of PTB-XL, 360 is the sample rate of MIT-BIH
        f,t, Zxx = stft(time_instance.transpose(1,0),fs=self.fs, window='hann',nperseg=self.nperseg)
        spectrogram_instance = np.abs(Zxx)  #(12, 63, 78)
        spectrogram_instance = spectrogram_instance.transpose(1,2,0)     #(63, 78, 12)
        return time_instance, spectrogram_instance
 
 
 
class TestSet(Dataset):
    def __init__(self, folder, fs=500, nperseg=125):
        self.test_data = np.load(os.path.join(folder, 'test.npy'))
        self.fs = fs
        self.nperseg = nperseg
       
    def __len__(self):
        return self.test_data.shape[0]
    
    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return np.array(peak_list)
   
    def __getitem__(self, index):
        time_instance = self.test_data[index]
        r_index = self.checkR(time_instance[:,1])
        time_instance = time_instance[100:4900,:]
        # Short Time Fast Fourier Transform
        f,t, Zxx = stft(time_instance.transpose(1,0),fs=self.fs, window='hann',nperseg=self.nperseg)
        spectrogram_instance = np.abs(Zxx)  #(12, 63, 78)
        spectrogram_instance = spectrogram_instance.transpose(1,2,0)     #(63, 78, 12)
        return time_instance, spectrogram_instance, r_index
 


# ---------------------------------------------------------------------------
# Diffusion branch (added alongside the original TSR-Net datasets).
#
# TrainSet/TestSet above are used unchanged by train.py and test.py.  The
# diffusion branch needs neither the STFT spectrogram nor the HeartPy R-peak
# list, and TestSet's variable-length `r_index` cannot be collated with a batch
# size > 1 anyway -- so it gets its own time-only dataset.
# ---------------------------------------------------------------------------

from diffusion.forward_process import (  # noqa: E402
    ECG_WINDOW_END,
    ECG_WINDOW_START,
    per_lead_minmax,
)


class DiffusionECGSet(Dataset):
    """Time-domain ECG windows for the diffusion branch.

    Yields ``(window, index)`` where ``window`` is ``[4800, 12]`` float32,
    channel-LAST -- the same orientation the original datasets use.  Callers
    convert to ``[B, 12, 4800]`` via ``diffusion.forward_process.to_channel_first``.

    normalize
        Per-record, per-lead min-max scaling to [-1, 1].  ``data/train.npy`` was
        saved WITH this transform applied (``preprocess.denoise_train``) but
        ``data/test.npy`` was saved WITHOUT it (``preprocess.denoise_test``), so
        the two files differ in amplitude by roughly 3.5x.  Diffusion SNR
        depends directly on the amplitude of x_0, so evaluating a
        train-distribution model on raw test amplitudes is not a valid
        comparison.  Set ``normalize=True`` for the test split to put both on
        the same footing; it is a no-op on already-normalised train data.
    """

    def __init__(self, folder, split='train', normalize=False, limit=None):
        if split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.path = os.path.join(folder, f'{split}.npy')
        # mmap: these files are 1-4 GB and only one window is touched at a time.
        self.data = np.load(self.path, mmap_mode='r')
        self.split = split
        self.normalize = normalize
        self.limit = limit if limit is None else min(int(limit), self.data.shape[0])

    def __len__(self):
        return self.data.shape[0] if self.limit is None else self.limit

    def __getitem__(self, index):
        window = np.asarray(self.data[index][ECG_WINDOW_START:ECG_WINDOW_END, :])  # (4800, 12)
        if self.normalize:
            window = per_lead_minmax(window)
        return window.astype(np.float32), index
