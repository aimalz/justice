import os
import os.path
import subprocess

from justice import path_util

time_series_dir = path_util.project_root / "time_series_demo"


def main():
    if not os.path.isdir(time_series_dir):
        subprocess.check_call([
            'git', 'clone', 'https://github.com/paztronomer/time_series_demo'
        ],
            cwd=os.path.dirname(time_series_dir))


if __name__ == '__main__':
    main()
