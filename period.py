import numpy as np
import concurrent.futures as confut
from itertools import chain

from justice.datasets import plasticc_data
from justice.features import period_distribution


def find_best_periods(lc):
    period_transform = period_distribution.MultiBandLs()
    mbp = period_transform.apply(lc)
    return (lc, mbp.best_periods, mbp.scores)


def main():
    with open("periodogram.csv", 'w') as f:
        source = plasticc_data.PlasticcBcolzSource.get_default()
        lcs_generator = plasticc_data.PlasticcDatasetLC.bcolz_get_all_lcs(
            source, 'training_set')
        with confut.ProcessPoolExecutor() as executor:
            for lc, best_periods, scores in executor.map(find_best_periods,
                                                         lcs_generator):
                print(",".join(map(str,
                                   [lc.meta['object_id'], lc.meta['ddf'],
                                    lc.meta['target'], *best_periods,
                                    *scores])), file=f)


if __name__ == '__main__':
    main()
