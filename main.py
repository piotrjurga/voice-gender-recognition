#!/usr/bin/python

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wav
from scipy.fft import fft

def classify(filename):
    sr, signal = wav.read(filename)
    if len(signal.shape) == 2:
        signal = signal[:,0]
    signal = signal/max(signal)

    freq_signal = abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/sr)
    #filter out frequencies outside of expected speech range
    mask = np.logical_and(freqs > 80, freqs < 250)
    freq_signal = freq_signal[mask]
    freqs = freqs[mask]
    #ranked = sorted(zip(freqs, freq_signal), key = lambda x: x[1], reverse = True)
    plt.clf()
    plt.plot(freqs, freq_signal)
    plt.savefig(f'{os.path.basename(filename)}.png')

    mean = sum(v*w for v,w in zip(freqs, freq_signal)) / sum(freq_signal)

    if mean < 180:
        res = 'M'
    else:
        res = 'K'
    print(filename, mean, res)
    return res

if not os.path.isdir('trainall'):
    print('test folder "trainall/" not found', file=sys.stderr)
    exit()

tests = 0
hits = 0
k_as_m = 0
m_as_k = 0

for f in os.listdir('trainall'):
    res = classify('trainall/' + f)
    tests += 1
    if f[4] == res: hits += 1
    if f[4] == 'K' and res == 'M':
        k_as_m += 1
        print('^ MISLABELED')
    if f[4] == 'M' and res == 'K':
        m_as_k += 1
        print('^ MISLABELED')
    #if tests > 16: break
print(f'tests: {tests}')
print(f'hits: {hits} ({hits/tests*100:.2f}%)')
print(f'k classified as m: {k_as_m}')
print(f'm classified as k: {m_as_k}')
