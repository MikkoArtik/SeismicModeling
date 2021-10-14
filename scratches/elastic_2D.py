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


layers = [(0, 150, 1.08, 0.54, 1.77, 16.6, 11.7),
          (150, 270, 2.5, 1.25, 2.19, 105, 74),
          (270, 690, 2.86, 1.43, 2.26, 141, 100),
          (690, 810, 3.33, 1.67, 2.35, 197, 139),
          (810, 890, 4, 2, 2.46, 295, 209),
          (890, 1050, 2.86, 1.43, 2.26, 141, 100),
          (1050, 1150, 5, 2.5, 2.6, 483, 341),
          (1150, 1170, 3.33, 1.67, 2.35, 197, 139),
          (1170, 1500, 4, 2, 2.46, 296, 209)]

# layer_type = 'empty'
# anomaly_layer = (900, 1000, 2.86, 1.43, 2.26, 141, 100)
#
# layer_type = 'oil'
# anomaly_layer = (900, 1000, 2.4, 1.2, 1.86, 1600, 80)

layer_type = 'water'
anomaly_layer = (900, 1000, 2.46, 1.23, 1.9, 3100, 80)

# layer_type = 'water+oil'
# anomaly_layer = (900, 1000, 2.43, 1.22, 1.9, 2350, 80)

extent = (1000, 1500)
delta_size = (10, 10)
t_start, t_stop = 0, 20000
dt = None

source_frequency = 500
space_order, time_order = 5, 1
nbl = 5

receiver_coords = np.array([[500, 0]])

source_coords = np.zeros(shape=(0, 2))
for i in range(0, 1010, 10):
    source_coords = np.vstack((source_coords, [i, 1500]))

shape = [int(x / delta_size[i]) + 1 for i, x in enumerate(extent)]

vp_model = np.zeros(shape=shape, dtype=np.float)
vs_model = np.zeros(shape=shape, dtype=np.float)
density_model = np.zeros(shape=shape, dtype=np.float)

for layer in layers[:-1]:
    top, bottom, vp_val, vs_val, density, qp_val, qs_val = layer
    top_index = int(top / delta_size[1])
    bottom_index = int(bottom / delta_size[1])
    vp_model[:, top_index: bottom_index] = vp_val
    vs_model[:, top_index: bottom_index] = vs_val
    density_model[:, top_index: bottom_index] = density

top, bottom, vp_val, vs_val, density, qp_val, qs_val = layers[-1]
top_index = int(top / delta_size[1])
vp_model[:, top_index:] = vp_val
vs_model[:, top_index:] = vs_val
density_model[:, top_index:] = density

if anomaly_layer:
    top, bottom, vp_val, vs_val, density, qp_val, qs_val = anomaly_layer
    top_index = int(top / delta_size[1])
    bottom_index = int(bottom / delta_size[1])
    vp_model[:, top_index:bottom_index] = vp_val
    vs_model[:, top_index:bottom_index] = vs_val
    density_model[:, top_index:bottom_index] = density

top_reflection = RightY2D()
top_reflection.set_parameters(model_spacing=delta_size, nbl=nbl,
                              model_shape=shape, position=0)
subdomains = (top_reflection,)

model = ModelElastic(shape=shape, spacing=delta_size,
                     space_order=space_order, vp=vp_model,
                     vs=vs_model, b=1/density_model,
                     nbl=nbl, origin=(0, 0),
                     subdomains=subdomains)

# model = ModelElastic(shape=shape, spacing=delta_size,
#                      space_order=space_order, vp=vp_model,
#                      vs=vs_model, b=1/density_model,
#                      nbl=nbl, origin=(0, 0))

if not dt:
    dt = model.critical_dt

source_frequency_kHz = source_frequency / 1000
geometry = AcquisitionGeometry(model=model, rec_positions=receiver_coords,
                               src_positions=source_coords, t0=t_start,
                               tn=t_stop, f0=source_frequency_kHz,
                               src_type='Ricker')
geometry._dt = dt
# plot_velocity(model, source=geometry.src_positions)
# plot_velocity(model)


source = geometry.src
rec_vx = Receiver(name='rec_vx', grid=model.grid,
                  time_range=geometry.time_axis,
                  coordinates=geometry.rec_positions)
rec_vz = Receiver(name='rec_vz', grid=model.grid,
                  time_range=geometry.time_axis,
                  coordinates=geometry.rec_positions)

lam, mu, irho, damp = model.lam, model.mu, model.b, model.damp

s = model.grid.stepping_dim.spacing

v = VectorTimeFunction(name='v', grid=model.grid, space_order=space_order,
                       time_order=time_order)
tau = TensorTimeFunction(name='t', grid=model.grid, space_order=space_order,
                         time_order=time_order)

# Particle velocity
u_v = Eq(v.forward, damp * (v + s * irho * div(tau)),
         subdomain=model.grid.subdomains['right_y'])
# Stress equations:
u_t = Eq(tau.forward, damp * tau + damp * dt * lam * diag(div(v.forward)) +
         damp * dt * mu * (grad(v.forward) + grad(v.forward).T),
         subdomain=model.grid.subdomains['right_y'])

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

op = Operator([u_v, u_t] + src_rec_expr, subs=model.grid.spacing_map)
op.apply(dt=dt)

# plot_image(v[1].data[1], cmap="seismic")

result_z = np.zeros(
    shape=(rec_vz.data.shape[0], receiver_coords.shape[0] + 1))
result_x = np.zeros(
    shape=(rec_vz.data.shape[0], receiver_coords.shape[0] + 1))

result_z[:, 0] = np.linspace(0, rec_vz.data.shape[0], rec_vz.data.shape[0])
result_z[:, 0] = result_z[:, 0]/(1000/dt)
result_x[:, 0] = result_z[:, 0]
for i in range(receiver_coords.shape[0]):
    result_z[:, i + 1] = rec_vz.data[:, i]
    result_x[:, i + 1] = rec_vx.data[:, i]

header = ['Time'] + [f'Dev_{x + 1}' for x in range(receiver_coords.shape[0])]
header = '\t'.join(header)
export_path = os.path.join(f'3d_vx-{layer_type}.dat')
np.savetxt(export_path, result_x, '%f', '\t', header=header, comments='')

export_path = os.path.join(f'3d_vz-{layer_type}.dat')
np.savetxt(export_path, result_z, '%f', '\t', header=header, comments='')