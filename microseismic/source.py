import sympy

from scipy import interpolate
from cached_property import cached_property
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    plt = None

from devito.types import Dimension, SparseTimeFunction

__all__ = ['PointSource', 'Receiver', 'Shot', 'WaveletSource',
           'RickerSource', 'GaborSource', 'DGaussSource', 'RandomSource',
           'PeriodicSource', 'MeanderSource', 'WhiteNoiseSource', 'TimeAxis']


class TimeAxis(object):
    """
    Data object to store the TimeAxis. Exactly three of the four key arguments
    must be prescribed. Because of remainder values it is not possible to create
    a TimeAxis that exactly adhears to the inputs therefore start, stop, step
    and num values should be taken from the TimeAxis object rather than relying
    upon the input values.

    The four possible cases are:
    start is None: start = step*(1 - num) + stop
    step is None: step = (stop - start)/(num - 1)
    num is None: num = ceil((stop - start + step)/step);
                 because of remainder stop = step*(num - 1) + start
    stop is None: stop = step*(num - 1) + start

    Parameters
    ----------
    start : float, optional
        Start of time axis.
    step : float, optional
        Time interval.
    num : int, optional
        Number of values (Note: this is the number of intervals + 1).
        Stop value is reset to correct for remainder.
    stop : float, optional
        End time.
    """

    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = step * (1 - num) + stop
            elif step is None:
                step = (stop - start) / (num - 1)
            elif num is None:
                num = int(np.ceil((stop - start + step) / step))
                stop = step * (num - 1) + start
            elif stop is None:
                stop = step * (num - 1) + start
            else:
                raise ValueError(
                    "Only three of start, step, num and stop may be set")
        except:
            raise ValueError(
                "Three of args start, step, num and stop may be set")

        if not isinstance(num, int):
            raise TypeError("input argument must be of type int")

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

    def __str__(self):
        return "TimeAxis: start=%g, stop=%g, step=%g, num=%g" % \
               (self.start, self.stop, self.step, self.num)

    def _rebuild(self):
        return TimeAxis(start=self.start, stop=self.stop, num=self.num)

    @cached_property
    def time_values(self):
        return np.linspace(self.start, self.stop, self.num)


class PointSource(SparseTimeFunction):
    """Symbolic data object for a set of sparse point sources

    Parameters
    ----------
    name : str
        Name of the symbol representing this source.
    grid : Grid
        The computational domain.
    time_range : TimeAxis
        TimeAxis(start, step, num) object.
    npoint : int, optional
        Number of sparse points represented by this source.
    data : ndarray, optional
        Data values to initialise point data.
    coordinates : ndarray, optional
        Point coordinates for this source.
    space_order : int, optional
        Space discretization order.
    time_order : int, optional
        Time discretization order (defaults to 2).
    dtype : data-type, optional
        Data type of the buffered data.
    dimension : Dimension, optional
        Represents the number of points in this source.
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})

        key = cls._cache_key(*args, **kwargs)
        obj = cls._cache_get(key)

        if obj is not None:
            newobj = sympy.Function.__new__(cls, *args, **options)
            newobj.__init_cached__(key)
            return newobj

        # Not in cache. Create a new PointSouce via devito.SparseTimeFunction

        name = kwargs.pop('name')
        grid = kwargs.pop('grid')
        time_range = kwargs.pop('time_range')
        time_order = kwargs.pop('time_order', 2)
        p_dim = kwargs.pop('dimension', Dimension(name='p_%s' % name))

        coordinates = kwargs.pop('coordinates',
                                 kwargs.pop('coordinates_data', None))
        # Either `npoint` or `coordinates` must be provided
        npoint = kwargs.pop('npoint', None)
        if npoint is None:
            if coordinates is None:
                raise TypeError("Need either `npoint` or `coordinates`")
            npoint = coordinates.shape[0]

        # Create the underlying SparseTimeFunction object
        obj = SparseTimeFunction.__new__(cls, name=name, grid=grid,
                                         dimensions=(grid.time_dim, p_dim),
                                         npoint=npoint, nt=time_range.num,
                                         time_order=time_order,
                                         coordinates=coordinates, **kwargs)

        obj._time_range = time_range._rebuild()

        # If provided, copy initial data into the allocated buffer
        data = kwargs.get('data')
        if data is not None:
            obj.data[:] = data

        return obj

    @cached_property
    def time_values(self):
        return self._time_range.time_values

    @property
    def time_range(self):
        return self._time_range

    def resample(self, dt=None, num=None, rtol=1e-5, order=3):
        # Only one of dt or num may be set.
        if dt is None:
            assert num is not None
        else:
            assert num is None

        start, stop = self._time_range.start, self._time_range.stop
        dt0 = self._time_range.step

        if dt is None:
            new_time_range = TimeAxis(start=start, stop=stop, num=num)
            dt = new_time_range.step
        else:
            new_time_range = TimeAxis(start=start, stop=stop, step=dt)

        if np.isclose(dt, dt0):
            return self

        nsamples, ntraces = self.data.shape

        new_traces = np.zeros((new_time_range.num, ntraces))

        for i in range(ntraces):
            tck = interpolate.splrep(self._time_range.time_values,
                                     self.data[:, i], k=order)
            new_traces[:, i] = interpolate.splev(new_time_range.time_values,
                                                 tck)

        # Return new object
        return PointSource(name=self.name, grid=self.grid, data=new_traces,
                           time_range=new_time_range,
                           coordinates=self.coordinates.data)

    # Pickling support
    _pickle_kwargs = SparseTimeFunction._pickle_kwargs + ['time_range']
    _pickle_kwargs.remove('nt')  # `nt` is inferred from `time_range`


Receiver = PointSource
Shot = PointSource


class WaveletSource(PointSource):
    """
    Abstract base class for symbolic objects that encapsulate a set of
    sources with a pre-defined source signal wavelet.

    Parameters
    ----------
    name : str
        Name for the resulting symbol.
    grid : Grid
        The computational domain.
    f0 : float
        Peak frequency for Ricker wavelet in kHz.
    time_values : TimeAxis
        Discretized values of time in ms.
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})

        key = cls._cache_key(*args, **kwargs)
        obj = cls._cache_get(key)

        if obj is not None:
            newobj = sympy.Function.__new__(cls, *args, **options)
            newobj.__init_cached__(key)
            return newobj

        # Not in cache. Create a new WaveletSouce via PointSource
        npoint = kwargs.pop('npoint', 1)
        obj = PointSource.__new__(cls, npoint=npoint, **kwargs)
        obj.f0 = kwargs.get('f0')
        for p in range(npoint):
            if cls is PeriodicSource:
                obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values,
                                             kwargs.get('package_frequency'))
            elif cls is RandomSource:
                obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values,
                                             kwargs.get('max_initial_count'),
                                             kwargs.get('frequency_type'))
            elif cls is MeanderSource:
                obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values,
                                             kwargs.get('amplitude'),
                                             kwargs.get('harm_count'),
                                             kwargs.get('t_offset'))
            elif cls is WhiteNoiseSource:
                obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values,
                                             kwargs.get('max_amplitude'),
                                             kwargs.get('t_start'),
                                             kwargs.get('t_stop'))
            else:
                obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values)

        return obj

    def wavelet(self, f0, t):
        """
        Defines a wavelet with a peak frequency f0 at time t.

        Parameters
        ----------
        f0 : float
            Peak frequency in kHz.
        t : TimeAxis
            Discretized values of time in ms.
        """
        raise NotImplementedError('Wavelet not defined')

    def show(self, idx=0, wavelet=None):
        """
        Plot the wavelet of the specified source.

        Parameters
        ----------
        idx : int
            Index of the source point for which to plot wavelet.
        wavelet : ndarray or callable
            Prescribed wavelet instead of one from this symbol.
        time : TimeAxis
            Prescribed time instead of time from this symbol.
        """
        wavelet = wavelet or self.data[:, idx]
        plt.figure()
        plt.plot(self.time_values, wavelet)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()

    # Pickling support
    _pickle_kwargs = PointSource._pickle_kwargs + ['f0']


class RickerSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:

    http://subsurfwiki.org/wiki/Ricker_wavelet

    Parameters
    ----------
    name : str
        Name for the resulting symbol.
    grid : Grid
        The computational domain.
    f0 : float
        Peak frequency for Ricker wavelet in kHz.
    time : TimeAxis
        Discretized values of time in ms.

    Returns
    ----------
    A Ricker wavelet.
    """

    def wavelet(self, f0, t):
        """
        Defines a Ricker wavelet with a peak frequency f0 at time t.

        Parameters
        ----------
        f0 : float
            Peak frequency in kHz.
        t : TimeAxis
            Discretized values of time in ms.
        """
        r = (np.pi * f0 * (t - 1. / f0))
        return (1 - 2. * r ** 2) * np.exp(-r ** 2) * 1e10


class GaborSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Gabor wavelet:

    https://en.wikipedia.org/wiki/Gabor_wavelet

    Parameters
    ----------
    name : str
        Name for the resulting symbol.
    grid : Grid
        defining the computational domain.
    f0 : float
        Peak frequency for Ricker wavelet in kHz.
    time : TimeAxis
        Discretized values of time in ms.

    Returns
    -------
    A Gabor wavelet.
    """

    def wavelet(self, f0, t):
        """
        Defines a Gabor wavelet with a peak frequency f0 at time t.
        Parameters
        ----------
        f0 : float
            Peak frequency in kHz.
        t : TimeAxis
            Discretized values of time in ms.
        """
        agauss = 0.5 * f0
        tcut = 1.5 / agauss
        s = (t - tcut) * agauss
        return np.exp(-2 * s ** 2) * np.cos(2 * np.pi * s)


class DGaussSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined 1st derivative wavelet of a Gaussian Source:

    Notes
    -----
    For visualizing the second or third order derivative
    of Gaussian wavelets, the convention is to use the
    negative of the normalized derivative. In the case
    of the second derivative, scaling by -1 produces a
    wavelet with its main lobe in the positive y direction.
    This scaling also makes the Gaussian wavelet resemble
    the Mexican hat, or Ricker, wavelet. The validity of
    the wavelet is not affected by the -1 scaling factor.

    Parameters
    ----------
    name : str
        Name for the resulting symbol.
    grid : Grid
        The computational domain.
    f0 : float
        Peak frequency for wavelet in kHz.
    time : TimeAxis
        Discretized values of time in ms.

    Returns
    ----------
    The 1st order derivative of the Gaussian wavelet
    """

    def wavelet(self, f0, t, a):
        """
        Defines the 1st derivative of a Gaussian wavelet.

        Parameters
        ----------
        f0 : float
            Peak frequency in kHz.
        t : TimeAxis
            Discretized values of time in ms.
        a : float
            Maximum amplitude.
        """
        return -2. * a * (t - 1 / f0) * np.exp(-a * (t - 1 / f0) ** 2)


class RandomSource(WaveletSource):
    def initial_signal(self, t0, f0, t):
        result = np.zeros(shape=t.shape[0])
        edge_index = np.argmax(t > t0)

        r = (np.pi * f0 * (t[edge_index:] - 1. / f0 - t[edge_index]))
        base_signal = (1 - 2. * r ** 2) * np.exp(-r ** 2)
        result[edge_index:] = base_signal
        return result

    def wavelet(self, f0, t, max_initial_count, frequency_type):
        initial_count = np.random.randint(1, 1 + max_initial_count)
        result_signal = np.zeros(shape=t.shape[0])
        unique_indexes = list()
        for i in range(initial_count):
            initial_time_index = np.random.randint(0, t.shape[0] - 1)
            if initial_time_index in unique_indexes:
                continue
            unique_indexes.append(initial_time_index)

            t0 = t[initial_time_index]
            if frequency_type == 'random':
                freq = np.random.rand() * f0
            else:
                freq = f0
            signal_i = self.initial_signal(t0=t0, f0=freq, t=t)
            result_signal = result_signal + signal_i

        return result_signal


class PeriodicSource(WaveletSource):
    @staticmethod
    def initial_signal(t0, f0, t):
        result = np.zeros(shape=t.shape[0])
        edge_index = np.argmax(t > t0)

        # print(edge_index, t0, t[edge_index])

        r = (np.pi * f0 * (t[edge_index:] - 1. / f0 - t[edge_index]))
        base_signal = (1 - 2. * r ** 2) * np.exp(-r ** 2)
        result[edge_index:] = base_signal
        return result

    def wavelet(self, f0, t, package_frequency):
        initial_count = int(t[-1] * package_frequency)
        result_signal = np.zeros(shape=t.shape[0])
        for i in range(initial_count):
            t0 = int(package_frequency * i * 1000000)
            signal_i = self.initial_signal(t0=t0, f0=f0, t=t)
            result_signal = result_signal + signal_i
        return result_signal


class MeanderSource(WaveletSource):

    def wavelet(self, f0, t, amplitude, harm_count, t_offset):
        result = np.zeros(shape=t.shape[0])
        edge_index = np.argmax(t > t_offset)
        for i in range(harm_count):
            k = 2 * i + 1
            result_i = np.zeros(shape=t.shape[0])
            result_i[edge_index:] = \
                np.sin(np.pi * f0 * (t[edge_index:] - t_offset) * k) / k
            result = result + result_i
        result = 4 * amplitude * result / np.pi
        result[t > t_offset + 1 / f0] = 0
        return result


class WhiteNoiseSource(WaveletSource):
    def wavelet(self, f0, t, max_amplitude, t_start, t_stop):
        result = np.zeros(shape=t.shape[0])
        f0_hz, f_recording_hz = 1000*f0, 1000/(t[1]-t[0])
        ratio=int(f_recording_hz/f0_hz)

        hit_count=int((t_stop-t_start)/f0)
        random_signal=max_amplitude*(2*np.random.rand(hit_count)-1)
        start_index=int(t_start/(t[1]-t[0]))
        for i in range(hit_count):
            left_index= start_index+ratio*i
            result[left_index:left_index+ratio]=random_signal[i]
        return result
