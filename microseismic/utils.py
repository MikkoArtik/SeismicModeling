from devito import error
from devito.tools import Pickable

from .source import *

__all__ = ['AcquisitionGeometry']


class AcquisitionGeometry(Pickable):
    """
    Encapsulate the geometry of an acquisition:
    - receiver positions and number
    - source positions and number

    In practice this would only point to a segy file with the
    necessary information
    """

    def __init__(self, model, rec_positions, src_positions, t0, tn,
                 **kwargs):
        """
        In practice would be __init__(segyfile) and all below parameters
        would come from a segy_read (at property call rather than at init)
        """
        self.rec_positions = rec_positions
        self._nrec = rec_positions.shape[0]
        self.src_positions = src_positions
        self._nsrc = src_positions.shape[0]
        self._src_type = kwargs.get('src_type')
        assert self.src_type in sources
        self._f0 = kwargs.get('f0')
        if self._src_type is not None and self._f0 is None:
            error("Peak frequency must be provided in KH" +
                  " for source of type %s" % self._src_type)

        self._model = model
        self._dt = model.critical_dt
        self._t0 = t0
        self._tn = tn

        self.package_frequency = kwargs.get('package_frequency')
        self.max_initial_count = kwargs.get('max_initial_count')
        self.frequency_type = kwargs.get('frequency_type')

        self.amplitude = kwargs.get('amplitude')
        self.harm_count = kwargs.get('harm_count')
        self.t_offset = kwargs.get('t_offset')

        self.max_amplitude = kwargs.get('max_amplitude')
        self.t_start = kwargs.get('t_start')
        self.t_stop = kwargs.get('t_stop')

    def resample(self, dt):
        self._dt = dt
        return self

    @property
    def time_axis(self):
        return TimeAxis(start=self.t0, stop=self.tn, step=self.dt)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def src_type(self):
        return self._src_type

    @property
    def grid(self):
        return self.model.grid

    @property
    def f0(self):
        return self._f0

    @property
    def tn(self):
        return self._tn

    @property
    def t0(self):
        return self._t0

    @property
    def dt(self):
        return self._dt

    @property
    def nt(self):
        return self.time_axis.num

    @property
    def nrec(self):
        return self._nrec

    @property
    def nsrc(self):
        return self._nsrc

    @property
    def dtype(self):
        return self.grid.dtype

    @property
    def rec(self):
        return Receiver(name='rec', grid=self.grid,
                        time_range=self.time_axis, npoint=self.nrec,
                        coordinates=self.rec_positions)

    @property
    def src(self):
        if self.src_type is None:
            return PointSource(name='src', grid=self.grid,
                               time_range=self.time_axis, npoint=self.nsrc,
                               coordinates=self.src_positions)
        elif self.src_type == 'Periodic':
            s = PeriodicSource(name='src', grid=self.grid, f0=self.f0,
                               time_range=self.time_axis, npoint=self.nsrc,
                               coordinates=self.src_positions,
                               package_frequency=self.package_frequency)
            return s
        elif self.src_type == 'Random':
            s = RandomSource(name='src', grid=self.grid, f0=self.f0,
                             time_range=self.time_axis, npoint=self.nsrc,
                             coordinates=self.src_positions,
                             max_initial_count=self.max_initial_count,
                             frequency_type=self.frequency_type)
            return s
        elif self.src_type == 'Meander':
            s = MeanderSource(name='src', grid=self.grid, f0=self.f0,
                              time_range=self.time_axis, npoint=self.nsrc,
                              coordinates=self.src_positions,
                              amplitude=self.amplitude,
                              harm_count=self.harm_count,
                              t_offset=self.t_offset)
            return s
        elif self.src_type == 'WhiteNoise':
            s = WhiteNoiseSource(name='src', grid=self.grid, f0=self.f0,
                                 time_range=self.time_axis, npoint=self.nsrc,
                                 coordinates=self.src_positions,
                                 max_amplitude=self.max_amplitude,
                                 t_start=self.t_start, t_stop=self.t_stop)
            return s
        else:
            return sources[self.src_type](name='src', grid=self.grid,
                                          f0=self.f0,
                                          time_range=self.time_axis,
                                          npoint=self.nsrc,
                                          coordinates=self.src_positions)

    _pickle_args = ['model', 'rec_positions', 'src_positions', 't0', 'tn']
    _pickle_kwargs = ['f0', 'src_type']


sources = {'Wavelet': WaveletSource, 'Ricker': RickerSource,
           'Gabor': GaborSource, 'Random': RandomSource,
           'Periodic': PeriodicSource, 'Meander': MeanderSource,
           'WhiteNoise': WhiteNoiseSource}
