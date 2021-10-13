import os

import numpy as np

from devito import *
from devito import Operator

from seismic import Receiver

from seismic import plot_velocity, plot_image

from microseismic.utils import AcquisitionGeometry
from microseismic.elastic.model import Model


class Elastic2DReflectionSolver:
    def __init__(self, model: Model):
        self.__model = model

        geometry = model.devito_geometry
        self.__rec_vx = Receiver(name='rec_vx', grid=model.devito_model.grid,
                                 time_range=model.devito_geometry.time_axis,
                                 coordinates=geometry.rec_positions)
        self.__rec_vz = Receiver(name='rec_vy', grid=model.devito_model.grid,
                                 time_range=model.devito_geometry.time_axis,
                                 coordinates=geometry.rec_positions)
        self.__velocity_field = VectorTimeFunction(
            name='v', grid=model.devito_model.grid,
            space_order=model.space_order, time_order=model.time_order)

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def geometry(self) -> AcquisitionGeometry:
        return self.model.devito_geometry

    @property
    def sensors_vx(self) -> Receiver:
        return self.__rec_vx

    @property
    def sensors_vz(self) -> Receiver:
        return self.__rec_vz

    @property
    def velocity_field(self) -> VectorTimeFunction:
        return self.__velocity_field

    def get_v_array(self, component='z') -> np.ndarray:
        if component.lower() == 'z':
            data = self.sensors_vz.data
        elif component.lower() == 'x':
            data = self.sensors_vx.data
        else:
            raise Exception('Invalid component name')

        discrete_count = data.shape[0]
        sensors_count = self.model.sensor_array.shape[0]
        result = np.zeros(shape=(discrete_count, sensors_count + 1))
        result[:, 0] = np.linspace(0, discrete_count, discrete_count)
        result[:, 0] *= self.model.time_space / 1000
        result[:, 0] = np.round(result[:, 0], 3)

        result[:-1, 1:] = data[:-1]
        return result

    def save_v_to_file(self, export_folder: str, component='z'):
        v_arr = self.get_v_array(component)

        header = ['Time'] + [f'Dev_{x + 1}' for x in
                             range(v_arr.shape[1] - 1)]
        header = '\t'.join(header)
        export_path = os.path.join(export_folder, f'2d_v{component}.dat')
        np.savetxt(export_path, v_arr, '%f', '\t', header=header,
                   comments='')

    def run(self):
        plot_velocity(self.model.devito_model,
                      source=self.model.sources_array,
                      receiver=self.model.sensor_array)
        lame = self.model.devito_model.lam
        mu = self.model.devito_model.mu
        inverse_density = self.model.devito_model.b
        damp = self.model.devito_model.damp
        s = self.model.devito_model.grid.stepping_dim.spacing

        v = self.velocity_field

        tau = TensorTimeFunction(name='t', grid=self.model.devito_model.grid,
                                 space_order=self.model.space_order,
                                 time_order=self.model.time_order)

        dt = self.model.time_space
        if self.model.is_reflection:
            u_v = Eq(v.forward, damp * (v + s * inverse_density * div(tau)),
                     subdomain=self.model.devito_model.grid.subdomains['right_y'])
            u_t = Eq(tau.forward,
                     damp * tau + damp * dt * lame * diag(div(v.forward)) +
                     damp * dt * mu * (grad(v.forward) + grad(v.forward).T),
                     subdomain=self.model.devito_model.grid.subdomains['right_y'])
        else:
            u_v = Eq(v.forward, damp * (v + s * inverse_density * div(tau)))
            u_t = Eq(tau.forward,
                     damp * tau + damp * dt * lame * diag(div(v.forward)) +
                     damp * dt * mu * (grad(v.forward) + grad(v.forward).T))

        source = self.model.devito_geometry.src

        src_xx = source.inject(field=tau.forward[0, 0], expr=source * s)
        src_zz = source.inject(field=tau.forward[1, 1], expr=source * s)
        src_expr = src_xx + src_zz

        rec_vx_e = self.sensors_vx.interpolate(expr=v[0])
        rec_vz_e = self.sensors_vz.interpolate(expr=v[1])
        rec_expr = rec_vx_e + rec_vz_e

        src_rec_expr = src_expr + rec_expr

        op = Operator([u_v, u_t] + src_rec_expr,
                      subs=self.model.devito_model.grid.spacing_map)
        op.apply(dt=dt)
        plot_image(v[1].data[1], cmap="seismic")
