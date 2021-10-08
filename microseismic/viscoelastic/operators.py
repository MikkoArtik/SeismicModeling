import sympy as sp

from examples.seismic import PointSource, Receiver

from devito import (Eq, Operator, VectorTimeFunction, TensorTimeFunction,
                    div, grad, diag)


def src_rec(v, tau, model, geometry):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid,
                      time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec1 = Receiver(name='rec1', grid=model.grid,
                    time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    rec2 = Receiver(name='rec2', grid=model.grid,
                    time_range=geometry.time_axis,
                    npoint=geometry.nrec)

    # The source injection term
    src_xx = src.inject(field=tau[0, 0].forward, expr=src * s)
    src_zz = src.inject(field=tau[-1, -1].forward, expr=src * s)
    src_expr = src_xx + src_zz
    if model.grid.dim == 3:
        src_yy = src.inject(field=tau[1, 1].forward, expr=src * s)
        src_expr += src_yy

    # Create interpolation expression for receivers
    rec_stress = rec1.interpolate(expr=tau[-1, -1])
    rec_flow = rec2.interpolate(expr=div(v))
    return src_expr + rec_stress + rec_flow


def src_rec_domains(v, tau, model, geometry):
    """
        Source injection and receiver interpolation
        """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid,
                      time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec_vx = Receiver(name='rec_vx', grid=model.grid,
                      time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vy = Receiver(name='rec_vy', grid=model.grid,
                      time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid,
                      time_range=geometry.time_axis,
                      npoint=geometry.nrec)

    # The source injection term
    src_xx = src.inject(field=tau[0, 0].forward, expr=src * s)
    src_yy = src.inject(field=tau[1, 1].forward, expr=src * s)
    src_zz = src.inject(field=tau[2, 2].forward, expr=src * s)
    src_expr = src_xx + src_yy + src_zz

    # Create interpolation expression for receivers
    rec_vx_e = rec_vx.interpolate(expr=v[0])
    rec_vy_e = rec_vy.interpolate(expr=v[1])
    rec_vz_e = rec_vz.interpolate(expr=v[2])
    rec_expr = rec_vx_e + rec_vy_e + rec_vz_e
    return src_expr + rec_expr


def create_operator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    l, qp, mu, qs, ro, damp = \
        model.lam, model.qp, model.mu, model.qs, model.irho, model.damp
    s = model.grid.stepping_dim.spacing

    f0 = geometry._f0
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    t_es = (1.+f0*qs*t_s)/(f0*qs-f0**2*t_s)

    # Create symbols for forward wavefield, source and receivers
    # Velocity:
    v = VectorTimeFunction(name="v", grid=model.grid,
                           save=geometry.nt if save else None,
                           time_order=1, space_order=space_order)
    # Stress:
    tau = TensorTimeFunction(name='t', grid=model.grid,
                             save=geometry.nt if save else None,
                             space_order=space_order, time_order=1)
    # Memory variable:
    r = TensorTimeFunction(name='r', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)

    # Particle velocity
    u_v = Eq(v.forward, damp * (v + s*ro*div(tau)))
    symm_grad = grad(v.forward) + grad(v.forward).T
    # Stress equations:
    u_t = Eq(tau.forward, damp * (r.forward + tau +
                                  s * (l * t_ep / t_s * diag(div(v.forward)) +
                                       mu * t_es / t_s * symm_grad)))

    # Memory variable equations:
    u_r = Eq(r.forward, damp * (r - s / t_s * (r + mu * (t_es/t_s-1) * symm_grad +
                                               l * (t_ep/t_s-1) * diag(div(v.forward)))))
    src_rec_expr = src_rec(v, tau, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator([u_v, u_r, u_t] + src_rec_expr, subs=model.spacing_map,
                    name='Forward', **kwargs)


def create_operator_domain(model, geometry, is_using_right_z_reflection,
                           space_order=4, save=False, **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    is_using_right_z_reflection: True or False
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    l, qp, mu, qs, ro, damp = \
        model.lam, model.qp, model.mu, model.qs, model.irho, model.damp
    s = model.grid.stepping_dim.spacing

    f0 = geometry._f0
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    t_es = (1.+f0*qs*t_s)/(f0*qs-f0**2*t_s)

    # Create symbols for forward wavefield, source and receivers
    # Velocity:
    v = VectorTimeFunction(name="v", grid=model.grid,
                           save=geometry.nt if save else None,
                           time_order=1, space_order=space_order)
    # Stress:
    tau = TensorTimeFunction(name='t', grid=model.grid,
                             save=geometry.nt if save else None,
                             space_order=space_order, time_order=1)
    # Memory variable:
    r = TensorTimeFunction(name='r', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)

    symm_grad = grad(v.forward) + grad(v.forward).T

    if is_using_right_z_reflection:
        u_v = Eq(v.forward, damp * (v + s*ro*div(tau)),
                 subdomain=model.grid.subdomains['right_z'])
        u_t = Eq(tau.forward, damp * (r.forward + tau + s * (l * t_ep /
                                 t_s * diag(div(v.forward)) + mu * t_es /
                                 t_s * symm_grad)),
                 subdomain=model.grid.subdomains['right_z'])
    else:
        u_v = Eq(v.forward, damp * (v + s * ro * div(tau)))
        u_t = Eq(tau.forward, damp * (r.forward + tau + s * (l * t_ep / t_s *
                              diag(div(v.forward)) + mu * t_es / t_s *
                              symm_grad)))

    u_r = Eq(r.forward, damp * (r - s / t_s * (r + mu * (t_es / t_s - 1) *
                        symm_grad + l * (t_ep / t_s - 1) * diag(
                        div(v.forward)))))

    src_rec_expr = src_rec_domains(v, tau, model, geometry)

    equations=[u_v, u_r, u_t]+src_rec_expr

    # Substitute spacing terms to reduce flops
    return Operator(equations, subs=model.spacing_map,
                    name='Forward', **kwargs)
