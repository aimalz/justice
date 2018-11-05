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
        lcs = chain.from_iterable(
            [plasticc_data.PlasticcDatasetLC.get_lcs_by_target(
                "data/plasticc_training_data.db", t, ddf=True)
             for t in (6, 53, 16, 65, 92, 15, 42, 52, 62, 64, 67, 88, 90, 95)])
        with confut.ProcessPoolExecutor() as executor:
            for lc, best_periods, scores in executor.map(find_best_periods,
                                                         lcs):
                print(",".join(map(str,
                                   [lc.meta['object_id'], lc.meta['ddf'],
                                    lc.meta['target'], *best_periods,
                                    *scores])), file=f)


if __name__ == '__main__':
    main()
