import os
from typing import List
import numpy as np

from microseismic.elastic.model import Model


def get_first_arrivals(signal_data: np.ndarray) -> List[float]:
    times = []
    for i in range(signal_data.shape[1] - 1):
        for j in range(signal_data.shape[0]):
            if signal_data[j, i + 1] != 0:
                times.append(signal_data[j, 0])
                break
    return times


def create_godograph(times: List[float],
                     distances: List[float]) -> np.ndarray:
    result = np.zeros(shape=(len(times), 2))
    result[:, 0] = distances
    result[:, 1] = times
    return result


if __name__ == '__main__':
    root = '/media/michael/Data/Projects/Ulyanovskoye_deposit/Modeling/VSP_Well_1617'
    model_filename = 'VSP_report.json'
    signals_filename = 'signals.dat'
    export_path = 'godograph.dat'

    signals_data = np.loadtxt(os.path.join(root, signals_filename),
                              skiprows=1, delimiter='\t')
    times = get_first_arrivals(signals_data)

    model = Model(os.path.join(root, model_filename), True)
    source_x_start = model.sources_x_start
    sensors_x_start = model.sensors_x_start
    sensor_space = model.sensors_space

    x_initial = sensors_x_start - source_x_start
    x_coords = [x_initial + x * sensor_space for x in range(len(times))]

    data = create_godograph(times, x_coords)

    header = 'distance\ttime_sec'
    np.savetxt(os.path.join(root, export_path), data, fmt='%f', header=header,
               comments='', delimiter='\t')





    print(times)
