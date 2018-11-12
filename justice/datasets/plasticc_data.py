# -*- coding: utf-8 -*-
"""Functions related to manipulating the PLAsTiCC dataset.

Note some functions are already in lightcurve.py's PlasticcDatasetLC class.
"""
import random
import pathlib
import sqlite3
import typing

import pandas as pd
import numpy as np

from justice import path_util
from justice import lightcurve
from justice.datasets import plasticc_bcolz
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


class PlasticcBcolzSource(object):
    _bcolz_cache = {}

    def __init__(self, bcolz_dir):
        self.bcolz_dir = bcolz_dir
        self.tables = {}

    def get_table(self, dataset):
        if dataset not in self.tables:
            bcolz_dataset = plasticc_bcolz.BcolzDataset(
                pathlib.Path(self.bcolz_dir) / dataset
            )
            self.tables[dataset] = bcolz_dataset.read_table()
        return self.tables[dataset]

    @classmethod
    def get_with_cache(cls, source):
        source = str(source)  # in case a pathlib.Path
        if source not in cls._bcolz_cache:
            cls._bcolz_cache[source] = PlasticcBcolzSource(source)
        return cls._bcolz_cache[source]

    @classmethod
    def get_default(cls):
        return cls.get_with_cache(plasticc_bcolz._root_dir)


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
    def bcolz_get_lcs_by_obj_ids(
        cls, bcolz_source: PlasticcBcolzSource, dataset: str, obj_ids: typing.List[int]
    ) -> typing.List['PlasticcDatasetLC']:
        bcolz_table = bcolz_source.get_table(dataset)
        meta_table = bcolz_source.get_table(dataset + '_metadata')
        if not obj_ids:
            return []
        query_parts = ['(object_id == {})'.format(obj_id) for obj_id in obj_ids]
        query = ' | '.join(query_parts)
        bcolz_map = bcolz_table.where(query)

        df = pd.DataFrame.from_records(bcolz_map, columns=bcolz_table.names)

        all_raw_bands = {}
        for group in df.groupby(['object_id', 'passband']):
            (object_id, passband), df_chunk = group
            if object_id not in all_raw_bands:
                all_raw_bands[object_id] = [None] * len(cls.expected_bands)
            all_raw_bands[object_id][passband] = lightcurve.BandData(
                np.array(df_chunk['mjd']),
                np.array(df_chunk['flux']),
                np.array(df_chunk['flux_err']),
                np.array(df_chunk['detected']),
            )
        lcs = {}
        for object_id, bands in all_raw_bands.items():
            assert None not in bands, "If raw data is missing whole bands, then we have to rethink things"
            lcs[object_id] = cls(**dict(zip(cls.expected_bands, bands)))

        bcolz_meta_map = meta_table.where(query)

        object_id_name_index = meta_table.names.index('object_id')
        for meta_row in bcolz_meta_map:
            obj_id = meta_row[object_id_name_index]
            lc = lcs[obj_id]
            lc.meta = {}
            for column_idx, column_name in enumerate(meta_table.names):
                lc.meta[column_name] = meta_row[column_idx]

        return list(lcs.values())

    @classmethod
    def get_lc(cls, source, dataset, obj_id):
        if isinstance(source, sqlite3.Connection):
            return cls._sqlite_get_lc(source, dataset, obj_id)
        elif isinstance(source, str) and source.endswith('.db'):
            with sqlite3.connect(source) as conn:
                return cls._sqlite_get_lc(conn, dataset, obj_id)
        elif isinstance(source, str) and 'plasticc_bcolz' in source:
            source = PlasticcBcolzSource.get_with_cache(source)
            maybe_lc = cls.bcolz_get_lcs_by_obj_ids(source, dataset, [obj_id])
            assert len(maybe_lc) == 1, "Did not find an LC of id {}".format(obj_id)
            return maybe_lc[0]
        elif isinstance(source, PlasticcBcolzSource):
            maybe_lc = cls.bcolz_get_lcs_by_obj_ids(source, dataset, [obj_id])
            assert len(maybe_lc) == 1, "Did not find an LC of id {}".format(obj_id)
            return maybe_lc[0]
        else:
            raise NotImplementedError(
                "Don't know how to read LCs from {}", format(source)
            )

    @classmethod
    def _sqlite_get_lcs_by_target(cls, conn, target):
        q = '''select object_id from training_set_meta where target = ?'''
        obj_ids = conn.execute(q, [target]).fetchall()
        return [cls._sqlite_get_lc(conn, 'training_set', o) for (o, ) in obj_ids]

    @classmethod
    def _bcolz_get_lcs_by_target(cls, source: PlasticcBcolzSource, target):
        bcolz_meta_table = source.get_table('training_set_metadata')
        bcolz_meta_map = bcolz_meta_table.where(
            'target == {}'.format(target), outcols=['object_id']
        )
        obj_ids = [row.object_id for row in bcolz_meta_map]

        return cls.bcolz_get_lcs_by_obj_ids(source, 'training_set', obj_ids)

    @classmethod
    def get_lcs_by_target(cls, source, target):
        # assuming training set because we don't have targets for the test set
        if isinstance(source, sqlite3.Connection):
            return cls._sqlite_get_lcs_by_target(source, target)
        elif isinstance(source, str) and source.endswith('.db'):
            with sqlite3.connect(source) as conn:
                return cls._sqlite_get_lcs_by_target(conn, target)
        elif isinstance(source, str) and 'plasticc_bcolz' in source:
            source = PlasticcBcolzSource.get_with_cache(source)
            return cls._bcolz_get_lcs_by_target(source, target)
        elif isinstance(source, PlasticcBcolzSource):
            return cls._bcolz_get_lcs_by_target(source, target)
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
