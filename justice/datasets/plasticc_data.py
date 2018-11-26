# -*- coding: utf-8 -*-
"""Functions related to manipulating the PLAsTiCC dataset.

Note some functions are already in lightcurve.py's PlasticcDatasetLC class.
"""
import bcolz
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
        self.bcolz_dir = pathlib.Path(bcolz_dir)
        self.tables = {}
        self.pandas_indices = {}  # (name, column) -> pd.DataFrame

    def get_table(self, table_name):
        if table_name not in self.tables:
            bcolz_dataset = plasticc_bcolz.BcolzDataset(self.bcolz_dir / table_name)
            self.tables[table_name] = bcolz_dataset.read_table()
        return self.tables[table_name]

    def get_pandas_index(self, table_name, column_name="object_id") -> pd.DataFrame:
        """Loads index table from bcolz to in-memory Pandas data frame.

        Takes a few hundred ms for ~3.5M rows, and a few fundred more on the first
        index lookup (use .loc[key]). Afterwards it should be super fast.

        :param table_name: Name of original bcolz table, e.g. "test_set"
        :param column_name: Name of column used as an index, usually "object_id"
        :return: DataFrame with the column set as its index.
        """
        index_dirname = self.bcolz_dir / f"{table_name}__{column_name}_index"
        if (table_name, column_name) not in self.pandas_indices:
            if not index_dirname.is_dir():
                raise EnvironmentError(
                    "Please run bcolz_generate_sorted_index (refer to "
                    "README_bcolz.md)."
                )
            table = bcolz.open(str(index_dirname))
            df = pd.DataFrame({k: table[k][:]
                               for k in set(table.names) - {column_name}},
                              index=table[column_name][:])
            self.pandas_indices[(table_name, column_name)] = df
        return self.pandas_indices[(table_name, column_name)]

    def all_object_ids(self, table_name):
        return self.get_pandas_index(table_name).index.values

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
        """Gets a list of light curves by object_id.

        :param bcolz_source: Data source instance, usually PlasticcBcolzSource.get_default().
        :param dataset: Name of the dataset, usually 'test_set' or 'training_set'.
        :param obj_ids: List of IDs. Should be unique, but seems to work OK otherwise.
        """
        index_table = bcolz_source.get_pandas_index(dataset)
        bcolz_table = bcolz_source.get_table(dataset)
        meta_table = bcolz_source.get_table(dataset + '_metadata')
        if not obj_ids:
            return []

        lcs = {}
        try:
            index_rows = index_table.loc[obj_ids].itertuples()
        except KeyError as e:
            raise KeyError(
                "Couldn't find requested object IDs in index! Original error: {!r}".
                format(e)
            )
        for object_id, index_row in zip(obj_ids, index_rows):
            subsel = bcolz_table[index_row.start_idx:index_row.end_idx]
            bands = {}
            for band_idx, band_name in enumerate(cls.expected_bands):
                passband_sel = subsel[subsel['passband'] == band_idx]
                bands[band_name] = lightcurve.BandData(
                    passband_sel['mjd'],
                    passband_sel['flux'],
                    passband_sel['flux_err'],
                    passband_sel['detected'],
                )
            lcs[object_id] = cls(**bands)

        meta_object_ids = meta_table['object_id'][:]
        meta_row_mask = np.isin(meta_object_ids, obj_ids, assume_unique=True)
        meta_rows = meta_table[meta_row_mask]

        object_id_name_index = meta_table.names.index('object_id')
        for meta_row in meta_rows:
            obj_id = meta_row[object_id_name_index]
            lc = lcs[obj_id]
            lc.meta = dict(zip(meta_table.names, meta_row))
        return list(lcs.values())

    @classmethod
    def bcolz_get_all_lcs(
        cls, bcolz_source: PlasticcBcolzSource, dataset: str, chunk_size=1000
    ) -> typing.Iterable['PlasticcDatasetLC']:
        """Generator that will eventually yield all light curves.

        :param bcolz_source: Data source.
        :param dataset: String typically 'training_set' or 'test_set'.
        :param chunk_size: Internal parameter determining how many light curves to retrieve
            at a time from bcolz.
        :yields: PLAsTiCC light curves.
        """
        ids_array = bcolz_source.all_object_ids(dataset)
        for start_idx in range(0, len(ids_array), chunk_size):
            ids_chunk = list(map(int, ids_array[start_idx:(start_idx + chunk_size)]))
            yield from cls.bcolz_get_lcs_by_obj_ids(
                bcolz_source, dataset=dataset, obj_ids=ids_chunk
            )

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
        return xform.BandNameMapper(
            **{
                'u': 368.55,
                'g': 484.45,
                'r': 622.95,
                'i': 753.55,
                'z': 868.6500000000001,
                'y': 967.8499999999999
            }
        )

    @classmethod
    def get_2dlcs_by_target(cls, source, target):
        lcs = cls.get_lcs_by_target(source, target)
        bnm = cls.get_bandnamemapper()
        return [bnm.make_lc2d(lc) for lc in lcs]
