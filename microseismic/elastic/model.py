from typing import Tuple
import json


import numpy as np

from seismic import ModelElastic

from microseismic.domains import RightY2D
from microseismic.utils import AcquisitionGeometry


class InvalidParameterName(Exception):
    pass


def load_model_file(path: str) -> dict:
    src_text = ''
    with open(path, 'r') as file_ctx:
        for line in file_ctx:
            src_text += line
    return json.loads(src_text)


def create_parameter_matrix(model: dict, param_name: str) -> np.ndarray:
    extent = (model['geometry']['x_extent'],
              model['geometry']['y_extent'])
    space = model['geometry']['space']

    shape = (int(extent[0] / space) + 1, int(extent[1] / space) + 1)

    param_array = np.zeros(shape=shape)
    for layer in model['model']:
        if param_name not in layer:
            raise InvalidParameterName
        top_edge, bottom_edge = layer['top'], layer['bottom']
        param_value = layer[param_name]

        min_index, max_index = int(top_edge / space), int(bottom_edge / space)
        param_array[:, min_index: max_index + 1] = param_value
    return param_array


class Model:
    def __init__(self, model_file: str, is_reflection=False):
        self.is_reflection = is_reflection
        self.__model_data = load_model_file(model_file)
        self.__devito_model = self.create_devito_model()
        self.__devito_geometry = self.create_devito_geometry()

    @property
    def src_data(self) -> dict:
        return self.__model_data

    @property
    def devito_model(self) -> ModelElastic:
        return self.__devito_model

    @property
    def devito_geometry(self) -> AcquisitionGeometry:
        return self.__devito_geometry

    @property
    def extent(self) -> Tuple[int, int]:
        x_extent = self.src_data['geometry']['x_extent']
        y_extent = self.src_data['geometry']['y_extent']
        return x_extent, y_extent

    @property
    def space(self) -> float:
        return self.src_data['geometry']['space']

    @property
    def space_xy(self) -> Tuple[float, float]:
        return self.space, self.space

    @property
    def shape(self) -> Tuple[int, int]:
        nx = int(self.extent[0] / self.space) + 1
        ny = int(self.extent[1] / self.space) + 1
        return nx, ny

    @property
    def source_depth(self) -> float:
        return self.src_data['source']['depth']

    @property
    def sources_array(self) -> np.ndarray:
        source_depth = self.source_depth
        sources = np.zeros(shape=(0, 2))
        for i in range(self.shape[0]):
            sources = np.vstack((sources, [i * self.space, source_depth]))
        return sources

    @property
    def sensor_array(self) -> np.ndarray:
        x = self.src_data['sensor']['x']
        y = self.src_data['sensor']['y']
        return np.array([[x, y]])

    @property
    def nbl(self) -> int:
        return self.src_data['parameters']['nbl']

    @property
    def space_order(self) -> int:
        return self.src_data['parameters']['space_order']

    @property
    def time_order(self) -> int:
        return self.src_data['parameters']['time_order']

    @property
    def reflection_edge(self) -> RightY2D:
        top_reflection = RightY2D()
        top_reflection.set_parameters(model_spacing=self.space_xy,
                                      nbl=self.nbl, model_shape=self.shape,
                                      position=0)
        return top_reflection

    def create_parameter_matrix(self, param_name: str) -> np.ndarray:
        param_array = np.zeros(shape=self.shape)
        for layer in self.src_data['model']:
            if param_name not in layer:
                raise InvalidParameterName
            top_edge, bottom_edge = layer['top'], layer['bottom']
            param_value = layer[param_name]

            min_index, max_index = int(top_edge / self.space), int(
                bottom_edge / self.space) + 1
            param_array[:, min_index: max_index] = param_value
        return param_array

    def create_devito_model(self) -> ModelElastic:
        vp = self.create_parameter_matrix('v_p') / 1000
        vs = self.create_parameter_matrix('v_s') / 1000
        density = self.create_parameter_matrix('density')
        if self.is_reflection:
            model = ModelElastic(space_order=4, vp=vp, vs=vs,
                                 b=1 / density, origin=(0, 0), shape=self.shape,
                                 spacing=self.space_xy, nbl=self.nbl,
                                 subdomains=(self.reflection_edge,))
        else:
            model = ModelElastic(space_order=4, vp=vp, vs=vs,
                                 b=1 / density, origin=(0, 0),
                                 shape=self.shape,
                                 spacing=self.space_xy, nbl=self.nbl)
        return model

    @property
    def time_space(self) -> float:
        recording_time_step_size = self.src_data['timeline']['dt'] * 1000
        if recording_time_step_size == 0:
            recording_time_step_size = self.devito_model.critical_dt
        return recording_time_step_size

    def create_devito_geometry(self) -> AcquisitionGeometry:
        t0 = self.src_data['timeline']['t0'] * 1000
        tn = self.src_data['timeline']['tn'] * 1000

        source_type = self.src_data['source']['type']
        source_frequency = self.src_data['source']['frequency'] / 1000

        geometry = AcquisitionGeometry(model=self.devito_model,
                                       rec_positions=self.sensor_array,
                                       src_positions=self.sources_array,
                                       t0=t0, tn=tn, src_type=source_type,
                                       f0=source_frequency)
        geometry._dt = self.time_space
        return geometry
