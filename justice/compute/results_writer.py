import json
import hashlib
import sqlite3
import os
import pathlib


def write_workresults_to_file(workresults: dict, directory: pathlib.Path, workresulttype: str) -> str:
    keys = [bytes(k, encoding='ascii') for k in workresults]
    filename = 'result-{}.json'.format(hashlib.md5(b''.join(keys)).hexdigest())
    os.makedirs(str(directory / workresulttype / 'chunks'), exist_ok=True)
    with open(directory / workresulttype / 'chunks' / filename, 'w') as f:
        json.dump(workresults, f, indent=2)
    return filename


def _read_workresults_from_file(filename: str) -> dict:
    with open(filename, 'r') as f:
        return json.load(f)


def _append_workresults_to_storage(workresults: dict, filename: pathlib.Path):
    sql_header = 'insert into WORKRESULTS (left_label, right_label, result) values '
    sql_footer = ', '.join(['(?, ?, ?)'] * len(workresults))
    values = []
    for labels_str, result in workresults.items():
        labels = labels_str.split(',')
        assert len(labels) == 2
        values.extend(labels)
        values.append(result)
    with sqlite3.connect(str(filename)) as conn:
        conn.execute(sql_header + sql_footer + ';', values)


def _init_storage_db(filename: pathlib.Path):
    with sqlite3.connect(str(filename)) as conn:
        conn.execute('''create table if not exists WORKRESULTS(
        left_label string,
        right_label string,
        result string, primary key(left_label, right_label));
        ''')


def scan_and_aggregate(directory: pathlib.Path, workresulttype: str) -> dict:
    _init_storage_db(directory / workresulttype / 'aggregated.db')
    workresult_files = os.listdir(directory / workresulttype / 'chunks')
    files_handled = {'unreadable': [],
                     'duplicate': [],
                     'invalid_key': [],
                     'ok': []}
    for wrf in workresult_files:
        try:
            workresults = _read_workresults_from_file(
                directory / workresulttype / 'chunks' / wrf)
        except:
            files_handled['unreadable'].append(wrf)
            continue
        try:
            _append_workresults_to_storage(
                workresults, directory / workresulttype / 'aggregated.db')
            os.remove(directory / workresulttype / 'chunks' / wrf)
            files_handled['ok'].append(wrf)
        except AssertionError:
            files_handled['invalid_key'].append(wrf)
            continue
        except:
            files_handled['duplicate'].append(wrf)
            continue
    return files_handled
