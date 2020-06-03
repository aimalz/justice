from justice.compute import results_writer
import argparse
import pathlib


def compare_two_lightcurve_ids(left_id, right_id):
    """uses fancymath to generate a goodness-of-merge score for two lightcurves"""
    if right_id == 0:
        return left_id
    elif right_id == 1:
        return 1
    elif right_id > left_id:
        return compare_two_lightcurve_ids(right_id, left_id)
    else:
        return compare_two_lightcurve_ids(right_id, left_id % right_id)


def get_lc_id_pairs():
    a = argparse.ArgumentParser()
    a.add_argument('lc_ids')
    d = a.parse_args()
    pairs = []
    for pair in d.split(';'):
        l, r = pair.split(',')
        pairs.append((int(l), int(r)))
    return pairs


def main():
    pairs = get_lc_id_pairs()
    workresults = {
        '{},{}'.format(pair[0], pair[1]): compare_two_lightcurve_ids(pair[0], pair[1])
        for pair in pairs
    }
    results_writer.write_workresults_to_file(
        workresults,
        pathlib.Path('/tmp/end-to-end-test'),
        'e2e'
    )


if __name__ == '__main__':
    main()
