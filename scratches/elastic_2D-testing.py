import os

import numpy as np
import sympy as sp

from devito import *
from devito import Operator

from seismic import Receiver, plot_velocity
from microseismic.utils import AcquisitionGeometry

from microseismic.domains import RightY2D
from microseismic.elastic.operators import src_rec_2d_with_velocity
from seismic import ModelElastic, plot_image

from microseismic.elastic.model import Model


model = Model('/media/michael/Data/Projects/ZapolarnoeDeposit/2021/Modeling'
              '/scratches/models/test_model.json', True)
dt = model.time_space
d_model = model.devito_model
d_geometry = model.devito_geometry

source = d_geometry.src
rec_vx = Receiver(name='rec_vx', grid=d_model.grid,
                  time_range=d_geometry.time_axis,
                  coordinates=d_geometry.rec_positions)
rec_vz = Receiver(name='rec_vz', grid=d_model.grid,
                  time_range=d_geometry.time_axis,
                  coordinates=d_geometry.rec_positions)

lam, mu, irho, damp = d_model.lam, d_model.mu, d_model.b, d_model.damp

s = d_model.grid.stepping_dim.spacing

v = VectorTimeFunction(name='v', grid=d_model.grid,
                       space_order=model.space_order,
                       time_order=model.time_order)
tau = TensorTimeFunction(name='t', grid=d_model.grid,
                         space_order=model.space_order,
                         time_order=model.time_order)

# Particle velocity
u_v = Eq(v.forward, damp * (v + s * irho * div(tau)),
         subdomain=d_model.grid.subdomains['right_y'])
# Stress equations:
u_t = Eq(tau.forward, damp * tau + damp * dt * lam * diag(div(v.forward)) +
         damp * dt * mu * (grad(v.forward) + grad(v.forward).T),
         subdomain=d_model.grid.subdomains['right_y'])

# # Particle velocity
# u_v = Eq(v.forward, damp * (v + s * irho * div(tau)))
# # Stress equations:
# u_t = Eq(tau.forward, damp * tau + damp * dt * lam * diag(div(v.forward)) +
#          damp * dt * mu * (grad(v.forward) + grad(v.forward).T))

src_xx = source.inject(field=tau.forward[0, 0], expr=source * s)
src_zz = source.inject(field=tau.forward[1, 1], expr=source * s)

src_expr = src_xx + src_zz

rec_vx_e = rec_vx.interpolate(expr=v[0])
rec_vz_e = rec_vz.interpolate(expr=v[1])
rec_expr = rec_vx_e + rec_vz_e

src_rec_expr = src_expr + rec_expr

op = Operator([u_v, u_t] + src_rec_expr, subs=d_model.grid.spacing_map)
op.apply(dt=dt)

# plot_image(v[1].data[1], cmap="seismic")

result_z = np.zeros(
    shape=(rec_vz.data.shape[0], model.sensor_array.shape[0] + 1))
result_x = np.zeros(
    shape=(rec_vz.data.shape[0], model.sensor_array.shape[0] + 1))

result_z[:, 0] = np.linspace(0, rec_vz.data.shape[0], rec_vz.data.shape[0])
result_z[:, 0] = result_z[:, 0]/(1000/dt)
result_x[:, 0] = result_z[:, 0]
for i in range(model.sensor_array.shape[0]):
    result_z[:, i + 1] = rec_vz.data[:, i]
    result_x[:, i + 1] = rec_vx.data[:, i]

header = ['Time'] + [f'Dev_{x + 1}' for x in range(model.sensor_array.shape[0])]
header = '\t'.join(header)
export_path = os.path.join(f'3d_vx.dat')
np.savetxt(export_path, result_x, '%f', '\t', header=header, comments='')

export_path = os.path.join(f'3d_vz.dat')
np.savetxt(export_path, result_z, '%f', '\t', header=header, comments='')
