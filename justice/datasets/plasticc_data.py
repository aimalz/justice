# -*- coding: utf-8 -*-
"""Functions related to manipulating the PLAsTiCC dataset.

Note some functions are already in lightcurve.py's PlasticcDatasetLC class.
"""
import os.path
import random

import sqlite3

import pandas as pd
import numpy as np

from justice import path_util
from justice import lightcurve
from justice import xform


def _filtered_fetch(df):
    """Filters out erroneous row filled with column names.

    TODO(gatoatigrado): Fix SQL ingestion or use of fetchall in the future.
    """
    df = df.copy()
    df = df[df['target'] != 'target']
    for key in ['object_id', 'target']:
        df[key] = pd.to_numeric(df[key])
    return df


class PlasticcDataset(object):
    def __init__(self, *, filename, base_table_name, seed=None):
        self.base_table_name = base_table_name
        self.rng = random.Random(seed)
        self.filename = filename
        self.conn = sqlite3.connect(filename)
        self.index_df = _filtered_fetch(
            pd.read_sql(
                sql="select object_id, target from {}_meta".format(base_table_name),
                con=self.conn,
            )
        )

    def __getitem__(self, key: str) -> pd.Series:
        return self.index_df.iloc[key]

    def get_lc(self, obj_id):
        from justice import lightcurve
        return PlasticcDatasetLC.get_lc(
            self.filename, self.base_table_name, obj_id=obj_id
        )

    def target_breakdown(self) -> pd.Series:
        """Returns the number of light curves for each classified type."""
        return self.index_df.groupby("target").size()

    def random_idx(self):
        return self.rng.randint(0, len(self.index_df) - 1)

    def random_lc(self):
        obj_id = int(self[self.random_idx()].object_id)
        return obj_id, self.get_lc(obj_id)

    @classmethod
    def training_data(cls):
        return cls(
            filename=str(path_util.data_dir / "plasticc_training_data.db"),
            base_table_name="training_set"
        )

    @classmethod
    def test_data(cls):
        return cls(
            filename=str(path_util.data_dir / "plasticc_test_data.db"),
            base_table_name="test_set"
        )


class PlasticcDatasetLC(lightcurve._LC):
    metadata_keys = [
        'object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz',
        'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target'
    ]

    expected_bands = list('ugrizy')

    @classmethod
    def _get_band_from_raw(cls, conn, dataset, obj_id, band_id):

        q = '''select mjd, flux, flux_err, detected
                from {}
                where object_id = ? and passband = ?
                order by mjd'''.format(dataset)
        cursor = conn.execute(q, [obj_id, band_id])
        times, fluxes, flux_errs, detected = [
            np.array(series) for series in zip(*cursor.fetchall())
        ]
        return lightcurve.BandData(times, fluxes, flux_errs, detected)

    @classmethod
    def _get_band_from_blobs(cls, conn, dataset, obj_id, band_id):
        res = conn.execute(
            '''
            select mjd, flux, flux_err, detected
            from {}_blob
            where object_id = ?
            and passband = ?
            '''.format(dataset), [obj_id, band_id]
        )
        raw_times, raw_fluxes, raw_flux_errs, raw_detected = res.fetchone()
        times = np.frombuffer(raw_times, dtype=np.float64)
        fluxes = np.frombuffer(raw_fluxes, dtype=np.float64)
        flux_errs = np.frombuffer(raw_flux_errs, dtype=np.float64)
        detected = np.frombuffer(raw_detected[::8], dtype=np.bool8)
        return lightcurve.BandData(times, fluxes, flux_errs, detected)

    @classmethod
    def _sqlite_get_lc(cls, conn, dataset, obj_id):
        has_blobs = conn.execute(
            "select name from sqlite_master where type='table' and "
            "name like '{}_blob'".format(dataset)
        ).fetchone()
        if has_blobs:
            loader = cls._get_band_from_blobs
        else:
            loader = cls._get_band_from_raw
        bands = dict((band, loader(conn, dataset, obj_id, band_id))
                     for band_id, band in enumerate(cls.expected_bands))
        lc = cls(**bands)

        meta_row = conn.execute(
            'select * from {}_meta where object_id = ?'.format(dataset), [obj_id]
        ).fetchone()
        lc.meta = dict(zip(cls.metadata_keys, meta_row))
        return lc

    @classmethod
    def get_lc(cls, source, dataset, obj_id):
        if isinstance(source, sqlite3.Connection):
            return cls._sqlite_get_lc(source, dataset, obj_id)
        elif isinstance(source, str) and source.endswith('.db'):
            with sqlite3.connect(source) as conn:
                return cls._sqlite_get_lc(conn, dataset, obj_id)
        else:
            raise NotImplementedError(
                "Don't know how to read LCs from {}", format(source)
            )

    @classmethod
    def _sqlite_get_lcs_by_target(cls, conn, target, ddf):
        q = '''select object_id from training_set_meta where target = ?'''
        if ddf:
            q += '''and ddf = 1'''
        obj_ids = conn.execute(q, [target]).fetchall()
        return [cls._sqlite_get_lc(conn, 'training_set', o) for (o, ) in obj_ids]

    @classmethod
    def get_lcs_by_target(cls, source, target, ddf=False):
        # assuming training set because we don't have targets for the test set
        if isinstance(source, sqlite3.Connection):
            return cls._sqlite_get_lcs_by_target(source, target, ddf)
        elif isinstance(source, str) and source.endswith('.db'):
            with sqlite3.connect(source) as conn:
                return cls._sqlite_get_lcs_by_target(conn, target, ddf)
        else:
            raise NotImplementedError(
                "Don't know how to read LCs from {}", format(source)
            )

    @classmethod
    def get_bandnamemapper(cls):
        # THESE ARE NOT THE REAL NUMBERS ALEX PLEASE FIX!! -davyd
        return xform.BandNameMapper(
            **{
                'u': 300.,
                'g': 400.,
                'r': 500.,
                'i': 600.,
                'z': 700.,
                'y': 800
            }
        )

    @classmethod
    def get_2dlcs_by_target(cls, source, target):
        lcs = cls.get_lcs_by_target(source, target)
        bnm = cls.get_bandnamemapper()
        return [bnm.make_lc2d(lc) for lc in lcs]
