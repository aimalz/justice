import abc
import base64
import collections
import io
import typing
import numpy as np
import scipy.stats as sps


class BandPoint(object):
    __slots__ = ('time', 'flux', 'flux_err', 'detected')

    def __init__(self, time, flux, flux_err, detected):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.detected = detected

    def __repr__(self):
        return (
            f"BandPoint[time={self.time}, flux={self.flux}, "
            f"flux_err={self.flux_err}, detected={self.detected}]"
        )

    def __eq__(self, other):
        return (
            self.time == other.time and self.flux == other.flux and
            self.flux_err == other.flux_err and self.detected == other.detected
        )


class BandData(object):
    """Light curve data for a single band.
    """

    __slots__ = ('time', 'flux', 'flux_err', 'detected')

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        detected: np.ndarray = None
    ) -> None:
        """Initializes BandData.

        :param time: Time values, 1-D np float array.
        :param flux: Flux values, 1-D np float array.
        :param flux_err: Flux error values, 1-D np float array.
        """
        assert time.shape == flux.shape == flux_err.shape

        if not np.issubdtype(time.dtype, np.floating):
            raise ValueError(f"Time must be floating array, got {time!r}")
        if not np.issubdtype(flux.dtype, np.floating):
            raise ValueError(f"Flux must be floating array, got {flux!r}")
        if not np.issubdtype(flux_err.dtype, np.floating):
            raise ValueError(f"Flux_err must be floating array, got {flux_err!r}")

        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        if detected is None:
            self.detected = np.ones_like(self.time)
        else:
            self.detected = detected

    def __repr__(self) -> str:
        """Formats light curve to a string for debugging."""
        return (
            'BandData(time={self.time}, flux={self.flux}, flux_err={'
            'self.flux_err})'
        ).format(self=self)

    def __add__(self, other: 'BandData') -> 'BandData':
        """Concatenates this light curve with another, and sorts by time.

        :param other: Other light curve band data.
        """
        # this function is a likely culprit for future slowness, given how many
        # times we'll be calling it.
        times = np.concatenate((self.time, other.time))
        fluxes = np.concatenate((self.flux, other.flux))
        flux_errs = np.concatenate((self.flux_err, other.flux_err))
        detecteds = np.concatenate((self.detected, other.detected))

        # tried kind='mergesort', but it wasn't any faster with 1e7 points
        ordinals = np.argsort(times)
        return BandData(
            times[ordinals], fluxes[ordinals], flux_errs[ordinals], detecteds[ordinals]
        )

    def _masked(self, mask: np.ndarray) -> 'BandData':
        return self.__class__(
            time=self.time[mask],
            flux=self.flux[mask],
            flux_err=self.flux_err[mask],
            detected=self.detected[mask],
        )

    def before_time(self, time: float, bias: float = 1e-8) -> 'BandData':
        """Gets all points strictly before 'time - bias'.

        :param time: Desired time to sample around, for windowed features.
        :param bias: Bias that can be positive, to exclude points close to the desired time,
            or negative, to include the desired time (and possibly points near it).
        :return: Instance of the same class, with some data masked.
        """
        return self._masked(self.time + bias < time)

    def after_time(self, time: float, bias: float = 1e-8) -> 'BandData':
        """Gets all points strictly after 'time + bias'.

        :param time: Desired time to sample around, for windowed features.
        :param bias: Bias that can be positive, to exclude points close to the desired time,
            or negative, to include the desired time (and possibly points near it).
        :return: Instance of the same class, with some data masked.
        """
        return self._masked(self.time > time + bias)

    def closest_point(self, time: float):
        idx = np.argmin(np.abs(self.time - time))
        return BandPoint(
            time=self.time[idx],
            flux=self.flux[idx],
            flux_err=self.flux_err[idx],
            detected=self.detected[idx],
        )

    @classmethod
    def from_cadence_shape_and_errfracs(cls, cadence, shape, errfracs):
        true_fluxes = shape(cadence)
        error_bars = errfracs * true_fluxes
        errors = sps.norm(0, error_bars).rvs(true_fluxes.shape)
        observed_fluxes = true_fluxes + errors
        return BandData(cadence, observed_fluxes, error_bars)

    def connect_the_dots(self) -> float:
        """Returns the arc length of the light curve.

        Sensitive to the magnitude of flux.

        :return: Arc length measurement.
        """
        # ignores errorbars
        dettimes = self.time[self.detected == 1]
        detfluxes = self.flux[self.detected == 1]
        time_diffs = dettimes[1:] - dettimes[:-1]
        flux_diffs = detfluxes[1:] - detfluxes[:-1]
        return float(np.sum(np.sqrt(time_diffs**2 + flux_diffs**2)))

    @classmethod
    def from_dense_array(cls, array: np.ndarray) -> 'BandData':
        num_points, channels = array.shape
        if channels != 3:
            raise ValueError("BandData.from_dense_array requires an n x 3 array.")
        return cls(time=array[:, 0], flux=array[:, 1], flux_err=array[:, 2])


class _LC:
    """Abstract base light curve class. Subclasses should provide a list of bands.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **bands: BandData) -> None:
        """Initializes a light curve.

        :param bands: Dictionary of bands.
        """
        if frozenset(bands.keys()) != frozenset(self.expected_bands):
            raise ValueError(
                "Expected bands {} but got {}".format(self.expected_bands, bands.keys())
            )

        d: collections.OrderedDict[str, BandData] = collections.OrderedDict()
        for b in self.expected_bands:
            d[b] = bands[b]
        for k in bands:
            assert k in self.expected_bands
        self.bands = d
        self.meta: dict = {}  # Free-form dict of metadata.

    def get_string_name(self) -> str:
        """Gets title for light curve; subclasses should use metadata."""
        return "Light curve with {} points".format(
            sum(len(band.time) for band in self.bands.values())
        )

    @property
    def nbands(self) -> int:
        """Returns the number of bands."""
        return len(self.bands)

    @property
    @abc.abstractmethod
    def expected_bands(self) -> typing.List[str]:
        """Returns list of expected bands.

        :return: List of expected bands
        :rtype: list[str]
        """
        raise NotImplementedError()

    def bands_sampled_in_region(self, time: float,
                                max_diff: float) -> typing.FrozenSet[str]:
        """Bands which have time points within (time - max_diff, time + max_diff).

        :param time: Time to query.
        :param max_diff: Max difference between actual sample time in band and `time`.
        :return: Frozenset of bands.
        """
        return frozenset(
            b for b, band_data in self.bands.items() if np.
            any((time - max_diff < band_data.time) & (band_data.time < time + max_diff))
        )

    def total_points_all_bands(self) -> int:
        return sum(len(band.time) for band in self.bands.values())

    def is_sane(self):
        """Put any check here that is too expensive at runtime, but useful for debugging"""
        sane = True
        for name, band in self.bands.items():
            band_is_sorted = (band.time[:-1] < band.time[1:]).all()
            sane = sane and band_is_sorted
        return sane

    def __repr__(self) -> str:
        kwargs = ', '.join([
            '{}={}'.format(band, data) for band, data in self.bands.items()
        ])
        return '{dataset}({kwargs})'.format(
            dataset=self.__class__.__name__, kwargs=kwargs
        )

    def _repr_html_(self):
        """Magic that lets us just type `lc` in jupyter and see a plot."""
        from justice import visualize
        from matplotlib import pyplot as plt
        fig = visualize.plot_lcs([self], title=self.get_string_name())
        tmpfile = io.BytesIO()
        fig.savefig(tmpfile, format='png')
        plt.close(fig)
        encoded = base64.b64encode(tmpfile.getvalue()).decode("ascii")
        return f"<img src='data:image/png;base64,{encoded}'>"

    def __add__(self, other: '_LC') -> '_LC':
        """Concatenates all bands of two light curves together.

        :param other: Other light curve.
        :return: New merged light curve.
        """
        assert self.expected_bands == other.expected_bands
        bands = {band: self.bands[band] + other.bands[band] for band in self.bands}
        return self.__class__(**bands)

    def __getitem__(self, key: str) -> BandData:
        return self.bands[key]

    def all_times(self) -> typing.List[np.ndarray]:
        return [bd.time for bd in self.bands.values()]

    def all_times_unique(self) -> np.ndarray:
        return np.unique(np.concatenate(self.all_times(), axis=0))

    def connect_the_dots(self) -> float:
        """Returns the sum of the arc length of all bands.

        :return: Arclength, summed over bands.
        """
        # ignores errorbars
        arclen = 0.
        for b in self.expected_bands:
            arclen += self.bands[b].connect_the_dots()
        return arclen


class LC2D:
    """Abstractly, a collection of points (plus errors) in (pwav, time, flux) space
    We hope to construct a 2D kernel to describe the distribution that these points
    are sampled from.

    The interface for `george` takes a single "x" and "y", which
    really mean "independent variables" and "dependent variables". So this class
    offers `invars` and `outvars` as arrays.
    """

    def __init__(self, pwav, time, flux, flux_err, detected):
        assert pwav.shape == time.shape
        assert time.shape == flux.shape
        assert flux.shape == flux_err.shape
        assert flux_err.shape == detected.shape
        self._invars = np.array([pwav, time])
        self._outvars = np.array([flux, flux_err])
        self._detected = detected

    @property
    def invars(self):
        return self._invars

    @property
    def outvars(self):
        return self._outvars

    @property
    def detected(self):
        return self._detected
