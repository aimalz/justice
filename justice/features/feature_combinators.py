# -*- coding: utf-8 -*-
"""Combinators for extraction functions."""
import typing

from justice import lightcurve


def combine(extract_fcns: typing.List):
    def helper(lc: lightcurve._LC, time: float):
        dicts = [fcn(lc, time) for fcn in extract_fcns]
        result = dicts[0]
        for dct in dicts[1:]:
            for k, v in dct.items():
                if k in result:
                    raise TypeError(
                        f"Multiple extract functions produced a key "
                        f"{k}.")
                result[k] = v
        return result

    return helper
