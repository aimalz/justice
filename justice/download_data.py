import subprocess
import os
import os.path

time_series_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../time_series_demo"))


def main():
    if not os.path.isdir(time_series_dir):
        subprocess.check_call(['git', 'clone', 'https://github.com/paztronomer/time_series_demo'],
                              cwd=os.path.dirname(time_series_dir))


if __name__ == '__main__':
    main()
