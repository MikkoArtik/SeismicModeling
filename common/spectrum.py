import numpy as np
from numpy.fft import rfft, rfftfreq


class Spectrum:
    def __init__(self, signal: np.ndarray, frequency: float):
        self.__signal = signal
        self.__frequency = frequency

        self.__spectrum_array = self.get()

    @property
    def signal(self) -> np.ndarray:
        return self.__signal

    @property
    def frequency(self) -> float:
        return self.__frequency

    @property
    def spectrum_array(self) -> np.ndarray:
        return self.__spectrum_array

    def get(self) -> np.ndarray:
        signal_count = self.signal.shape[0]
        spectrum_data = rfft(self.signal - np.mean(self.signal))
        res = np.empty((signal_count // 2 + 1, 2), dtype=np.float)
        res[:, 0] = rfftfreq(signal_count, 1 / self.frequency)
        res[:, 1] = 2 * abs(spectrum_data) / signal_count
        return res

    def save(self, output_path: str):
        header = 'freq\tamp'
        np.savetxt(output_path, self.spectrum_array, '%f', '\t', comments='',
                   header=header)
