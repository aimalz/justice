# -*- coding: utf-8 -*-
"""Declares builder of OGLE data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import re

import pandas as pd

from justice import ogle_data

# NOTE(gatoatigrado): Add new filename matches as necessary.
_extract_name = re.compile(r"(OGLE-LMC-CEP-\d+).dat")


def _dat_files(directory):
    matches = [_extract_name.match(filename) for filename in os.listdir(directory)]
    return [(match.group(0), match.group(1)) for match in matches]


def parse_lc_line(line):
    hjd, mag, mag_err = map(float, line.strip().split())
    flux = 10.0**23.0 * 10.0**(-0.4 * (mag + 48.6))
    flux_err = (mag_err / mag) * flux
    return hjd, flux, flux_err


def build(ogle3_data_dir, ia_descriptor, expected_bands=('I', 'V'), star_table=None):
    """Builds a binary OGLE dataset from original .DAT files.

    :param ogle3_data_dir: OGLE-3 data dir.
    :type ogle3_data_dir: str
    :param ia_descriptor: Descriptor of where to save data.
    :type ia_descriptor: justice.mmap_array.IndexedArrayDescriptor
    :param expected_bands: Expected bands.
    :param star_table: Star table filename.
    """

    if star_table:
        star_table = pd.read_csv(star_table, sep="\t").set_index("ID")

    band_dirs = [os.path.join(ogle3_data_dir, band) for band in expected_bands]
    for band_dir in band_dirs:
        if not os.path.isdir(band_dir):
            raise OSError("Expected directory {} to exist.".format(band_dir))

    band_names = sorted(
        frozenset.
        intersection(*[frozenset(_dat_files(band_dir)) for band_dir in band_dirs])
    )

    if not band_names:
        raise ValueError("Expected filenames to be shared.")

    id_str_to_info = {}
    if star_table is not None:
        for filename, id_str in band_names:
            row = star_table.ix[id_str
                                ]  # If KeyError, does not contain info for all files.
            id_str_to_info[id_str] = {
                "field": row["Field"],
                "StarID": row["StarID"],
                "Type": row["Type"],
                "Subtype": row["Subtype"],
            }

    index_rows = []
    data = []
    for filename, id_str in band_names:
        for band, band_dir in zip(expected_bands, band_dirs):
            index_rows.append(dict(
                id_str_to_info[id_str],
                id=id_str,
                band=band,
            ))
            with open(os.path.join(band_dir, filename)) as f:
                contents = list(map(parse_lc_line, f.read().strip().splitlines()))
            data.append(contents)

    ia_descriptor.write(index_rows, data, set_index='id')
    if "pytest" not in ia_descriptor.index_filename:
        print(
            "Wrote dataset in {} with {} light curves".format(
                ia_descriptor.index_filename, len(data)
            )
        )


def main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--name", required=True)
    cmd_args.add_argument("--data-directory", required=True)
    cmd_args.add_argument("--expected-bands", default="I,V", help="Expected band names.")
    cmd_args.add_argument("--star-table")
    args = cmd_args.parse_args()

    expected_bands = tuple(args.expected_bands.split(','))
    ia_descriptor = ogle_data.for_subset(args.name)
    build(
        args.data_directory,
        ia_descriptor,
        expected_bands=expected_bands,
        star_table=args.star_table
    )


if __name__ == '__main__':
    main()
