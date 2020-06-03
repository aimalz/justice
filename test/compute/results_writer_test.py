from justice.compute import results_writer
import pathlib
import os
import shutil


def test_results_writer():
    path = pathlib.Path('/tmp/justice_results_writer_test')
    if os.path.exists(path):
        shutil.rmtree(path)
    results_writer.write_workresults_to_file(
        {
            '613,718': '[1,1,0,0]',
            '718,945': '[2,1,0.5,3]'
        },
        path,
        'mergeparams'
    )
    results_writer.write_workresults_to_file(
        {
            'invalid_key!': '[1,1,0,0]',
        },
        path,
        'mergeparams'
    )
    results_writer.write_workresults_to_file(
        {
            '614,718': '[1,1,0,0]',
            '717,945': '[2,1,0.5,3]'
        },
        path,
        'mergeparams'
    )
    files_handled = results_writer.scan_and_aggregate(path, 'mergeparams')
    assert len(files_handled['unreadable']) == 0
    assert len(files_handled['duplicate']) == 0
    assert len(files_handled['invalid_key']) == 1
    assert len(files_handled['ok']) == 2
