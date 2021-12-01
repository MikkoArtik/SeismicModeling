import os

import numpy as np

from microseismic.elastic.model import Model
from microseismic.elastic.solver import Elastic2DReflectionSolver
from common.spectrum import Spectrum


if __name__ == '__main__':
    root_folder = '/media/michael/Data/Projects/Ulyanovskoye_deposit/' \
                  'Modeling/VSP_Well_1617'
    file_path = 'VSP_report.json'
    export_signal_filename = 'signals.dat'

    model = Model(os.path.join(root_folder, file_path), is_reflection=True)

    solver = Elastic2DReflectionSolver(model)
    solver.run()

    t_min, t_max = model.spectrum_time_limits
    signal = solver.get_v_array('z')

    header = ['Time']
    for i in range(model.sensors_count):
        sensor_x = i * model.sensors_space
        header.append(f'Sensor_x={sensor_x}')

    header_line = '\t'.join(header)
    export_path = os.path.join(root_folder, export_signal_filename)
    np.savetxt(export_path, signal, '%f', '\t', header=header_line,
               comments='')
