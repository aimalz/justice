# -*- coding: utf-8 -*-
"""Metadata feature extractor."""
from justice import lightcurve


class MetadataValueExtractor(object):
    def __init__(self):
        pass

    def extract(self, lc: lightcurve._LC, time: float):
        del time  # unused
        return {"object_id": lc.meta["object_id"]}
