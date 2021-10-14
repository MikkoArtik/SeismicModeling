import os

import numpy as np
from numpy.fft import rfft, rfftfreq


def spectrum(signal: np.ndarray, frequency: int) -> np.ndarray:
    """
    Method for calculating simple Fourier spectrum of signal
    :param signal: input signal
    :param frequency: signal frequency
    :return: 2D array of spectrum data
    """
    signal_count = signal.shape[0]
    spectrum_data = rfft(signal-np.mean(signal))
    res = np.empty((signal_count // 2 + 1, 2), dtype=float)
    res[:, 0] = rfftfreq(signal_count, 1 / frequency)
    res[:, 1] = 2 * abs(spectrum_data) / signal_count
    return res


if __name__ == '__main__':
    root = '/media/michael/Data/Projects/ZapolarnoeDeposit/2021/Modeling' \
           '/scratches'
    signal_file = '3d_vz-water.dat'
    sp_file = '3d_vz_spectrum.dat'
    data = np.loadtxt(os.path.join(root, signal_file), skiprows=1,
                                   delimiter='\t')
    freq = 825.08
    # signal = data[data[:, 0] > 250, 1]
    sp_data = spectrum(data[:, 1], freq)
    np.savetxt(os.path.join(root, sp_file), sp_data, '%f', '\t',
               header='Freq\tAmp', comments='')
