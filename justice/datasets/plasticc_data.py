# -*- coding: utf-8 -*-
"""Functions related to manipulating the PLAsTiCC dataset.

Note some functions are already in lightcurve.py's PlasticcDatasetLC class.
"""
import os.path
import random
import pathlib
import sqlite3

import pandas as pd
import numpy as np

from justice import path_util
from justice import lightcurve
from justice.datasets import plasticc_bcolz


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
        return lightcurve.PlasticcDatasetLC.get_lc(
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
    def _bcolz_get_lcs(cls, bcolz_table, meta_table, obj_ids):
        query_parts = ['(object_id == {})'.format(o) for o in obj_ids]
        query = ' | '.join(query_parts)
        bcolz_map = bcolz_table.where(query)
        # TODO: make this more efficient
        all_raw_bands = {}
        for o in obj_ids:
            all_raw_bands[o] = {}
            for passband, _ in enumerate(cls.expected_bands):
                # ('time', 'flux', 'flux_err', 'detected')
                all_raw_bands[o][passband] = ([], [], [], [])
        for row in bcolz_map:
            o = row[0]
            passband = row[2]
            all_raw_bands[o][passband][0].append(row[1])  # 'mjd'
            all_raw_bands[o][passband][1].append(row[3])  # 'flux'
            all_raw_bands[o][passband][2].append(row[4])  # 'flux_err'
            all_raw_bands[o][passband][3].append(row[5])  # 'detected'
        lcs = {}
        for o, raw_bands in all_raw_bands.items():
            bands = {}
            for passband, (time, flux, flux_err, detected) in raw_bands.items():
                bands[cls.expected_bands[passband]] = lightcurve.BandData(
                    np.array(time),
                    np.array(flux),
                    np.array(flux_err),
                    np.array(detected))
            lcs[o] = cls(**bands)
        meta_attrs = ['ddf', 'decl', 'distmod', 'gal_b', 'gal_l', 'hostgal_photoz',
                      'hostgal_photoz_err', 'hostgal_specz', 'mwebv', 'object_id', 'ra']

        bcolz_meta_map = meta_table.where(query)

        for meta_row in bcolz_meta_map:
            o = meta_row[0]
            lc = lcs[o]
            lc.meta = {}
            for m in meta_attrs:
                lc.meta[m] = getattr(meta_row, m)
            if hasattr(meta_row, 'target'):
                lc.meta['target'] = getattr(meta_row, 'target')

        return list(lcs.values())

    @classmethod
    def get_lc(cls, source, dataset, obj_id):
        if isinstance(source, sqlite3.Connection):
            return cls._sqlite_get_lc(source, dataset, obj_id)
        elif isinstance(source, str) and source.endswith('.db'):
            with sqlite3.connect(source) as conn:
                return cls._sqlite_get_lc(conn, dataset, obj_id)
        elif isinstance(source, str) and 'plasticc_bcolz' in source:
            bcolz_dataset = plasticc_bcolz.BcolzDataset(pathlib.Path(source) / dataset)
            bcolz_table = bcolz_dataset.read_table()
            bcolz_meta = plasticc_bcolz.BcolzDataset(pathlib.Path(source) / (dataset + '_meta'))
            meta_table = bcolz_meta.read_table()
            return cls._bcolz_get_lcs(bcolz_table, meta_table, [obj_id])[0]
        else:
            raise NotImplementedError(
                "Don't know how to read LCs from {}", format(source)
            )

    @classmethod
    def _sqlite_get_lc_by_target(cls, conn, target):
        q = '''select object_id from training_set_meta where target = ?'''
        obj_ids = conn.execute(q, [target]).fetchall()
        return [cls._sqlite_get_lc(conn, 'training_set', o) for (o, ) in obj_ids]

    @classmethod
    def _bcolz_get_lc_by_target(cls, source, target):
        bcolz_meta = plasticc_bcolz.BcolzDataset(pathlib.Path(source) / 'training_set_meta')
        bcolz_meta_table = bcolz_meta.read_table()
        bcolz_meta_map = bcolz_meta_table.where('target == {}'.format(target), outcols=['object_id'])
        obj_ids = [row.object_id for row in bcolz_meta_map]

        bcolz_dataset = plasticc_bcolz.BcolzDataset(pathlib.Path(source) / 'training_set')
        bcolz_table = bcolz_dataset.read_table()

        return cls._bcolz_get_lcs(bcolz_table, bcolz_meta_table, obj_ids)

    @classmethod
    def get_lc_by_target(cls, source, target):
        # assuming training set because we don't have targets for the test set
        if isinstance(source, sqlite3.Connection):
            return cls._sqlite_get_lc_by_target(source, target)
        elif isinstance(source, str) and source.endswith('.db'):
            with sqlite3.connect(source) as conn:
                return cls._sqlite_get_lc_by_target(conn, target)
        elif isinstance(source, str) and 'plasticc_bcolz' in source:
            return cls._bcolz_get_lc_by_target(source, target)
        else:
            raise NotImplementedError(
                "Don't know how to read LCs from {}", format(source)
            )
