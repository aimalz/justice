# -*- coding: utf-8 -*-
"""Declares builder of OGLE data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import os
import os.path

from justice import ogle_data


# NOTE(gatoatigrado): Add new filename matches as necessary.
_extract_name = re.compile(r"(OGLE-LMC-CEP-\d+).dat")


def _dat_files(directory):
    matches = [_extract_name.match(filename) for filename in os.listdir(directory)]
    return [(match.group(0), match.group(1)) for match in matches]


def build(ogle3_data_dir, ia_descriptor, expected_bands=('I', 'V')):
    """Builds a binary OGLE dataset from original .DAT files.

    :param ogle3_data_dir: OGLE-3 data dir.
    :type ogle3_data_dir: str
    :param ia_descriptor: Descriptor of where to save data.
    :type ia_descriptor: justice.mmap_array.IndexedArrayDescriptor
    :param expected_bands: Expected bands.
    """
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

    index_rows = []
    data = []
    for filename, id_str in band_names:
        for band, band_dir in zip(expected_bands, band_dirs):
            index_rows.append({
                'id': id_str,
                'band': band,
            })
            with open(os.path.join(band_dir, filename)) as f:
                contents = [
                    list(map(float,
                             line.strip().split()))
                    for line in f.read().strip().splitlines()
                ]
            data.append(contents)

    ia_descriptor.write(index_rows, data)
    print("Wrote dataset in {} with {} light curves".format(
        ia_descriptor.index_filename, len(data)))


def main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument("--name", required=True)
    cmd_args.add_argument("--data-directory", required=True)
    cmd_args.add_argument("--expected-bands", default="I,V", help="Expected band names.")
    args = cmd_args.parse_args()

    expected_bands = tuple(args.expected_bands.split(','))
    ia_descriptor = ogle_data.for_subset(args.name)
    build(args.data_directory, ia_descriptor, expected_bands=expected_bands)


if __name__ == '__main__':
    main()
