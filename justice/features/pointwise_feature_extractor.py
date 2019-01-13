# -*- coding: utf-8 -*-
"""Base class for per-light-curve feature extractors."""
import abc
import typing

from justice import lightcurve


class PointwiseFeatureExtractor(metaclass=abc.ABCMeta):
    """Base class for feature extractors which operate on a light curve and time.

    The alignment model tries to generate feature vectors for each point, so technically
    this is a bit more general, since the `time` value provided does not have to line
    up with any individual point.
    """

    @abc.abstractmethod
    def extract(self, lc: lightcurve._LC, time: float) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError()
