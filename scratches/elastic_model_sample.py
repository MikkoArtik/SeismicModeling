from typing import List, Tuple, NamedTuple
import json

import numpy as np

from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag

from seismic.model import ModelElastic
from seismic import Receiver
from seismic import plot_image

from microseismic.domains import RightY2D
from microseismic.utils import AcquisitionGeometry
from microseismic.elastic.model import Model
from microseismic.elastic.operators import get_reflection_2d_forward_operator
from microseismic.elastic.wavesolver import ElasticWaveSolver2DVersion


def run_test_modelling():
    model_file = 'models/test_model.json'
    model = Model(model_file)

    d_model = model.devito_model
    d_geometry = model.geo
    model_data = load_model_file(model_file)

    extent = (model_data['geometry']['x_extent'],
              model_data['geometry']['y_extent'])
    space = model_data['geometry']['space']

    shape = (int(extent[0] / space) + 1, int(extent[1] / space) + 1)

    source_depth = model_data['source']['depth']
    sources = np.zeros(shape=(shape[0], 2))
    for i in range(shape[0]):
        sources[i] = [space * i, source_depth]

    sensor = np.array([[model_data['sensor']['x'],
                          model_data['sensor']['y']]])

    v_p_model = create_parameter_matrix(model_data, 'v_p') / 1000
    v_s_model = create_parameter_matrix(model_data, 'v_s') / 1000
    b_model = 1 / create_parameter_matrix(model_data, 'density')

    top_reflection = RightY2D()
    top_reflection.set_parameters(model_spacing=(space, space), nbl=5,
                                  model_shape=shape, position=0)

    subdomains = (top_reflection, )
    model = ModelElastic(space_order=4, vp=v_p_model, vs=v_s_model,
                         b=b_model, origin=(0, 0), shape=shape,
                         spacing=(space, space), nbl=10,
                         subdomains=subdomains)

    t0 = model_data['timeline']['t0'] * 1000
    tn = model_data['timeline']['tn'] * 1000
    source_type = model_data['source']['type']
    source_frequency = model_data['source']['frequency'] / 1000
    geometry = AcquisitionGeometry(model=model, rec_positions=sensor,
                                   src_positions=sources, t0=t0, tn=tn,
                                   src_type=source_type, f0=source_frequency)

    recording_time_step_size = model_data['timeline']['dt']
    if recording_time_step_size == 0:
        recording_time_step_size = model.critical_dt
    else:
        recording_time_step_size *= 1000
    model.dt_scale = 0.9
    geometry._dt = recording_time_step_size

    source = geometry.src
    rec_vx = Receiver(name='rec_vx', grid=model.grid,
                      time_range=geometry.time_axis,
                      coordinates=geometry.rec_positions)
    rec_vz = Receiver(name='rec_vz', grid=model.grid,
                      time_range=geometry.time_axis,
                      coordinates=geometry.rec_positions)

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=False, space_order=4, time_order=1)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             save=False, space_order=4, time_order=1)

    lam, mu, b = model.lam, model.mu, model.b

    dt = model.critical_dt
    u_v = Eq(v.forward, model.damp * v + model.damp * dt * b * div(tau),
             subdomain=subdomains)
    u_t = Eq(tau.forward, model.damp * tau +
             model.damp * dt * lam * diag(div(v.forward)) +
             model.damp * dt * mu * (grad(v.forward) + grad(v.forward).T),
             subdomain=subdomains)

    s = model.grid.stepping_dim.spacing

    src_xx = source.inject(field=tau.forward[0, 0], expr=source * s)
    src_zz = source.inject(field=tau.forward[1, 1], expr=source * s)

    src_expr = src_xx + src_zz

    rec_vx_e = rec_vx.interpolate(expr=v[0])
    rec_vz_e = rec_vz.interpolate(expr=v[1])
    rec_expr = rec_vx_e + rec_vz_e

    src_rec_expr = src_expr + rec_expr
    op = Operator([u_v, u_t] + src_rec_expr, subs=model.grid.spacing_map)
    op.apply()

    plot_image(v[1].data[1], cmap="seismic")


if __name__ == '__main__':
    run_test_modelling()

