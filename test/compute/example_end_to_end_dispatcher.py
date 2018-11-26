from justice.compute import results_writer
import random
import subprocess

ROOT = '/home/aim267/justice'


SL_TEMPLATE = f'''
    #!/bin/bash -l
    #SBATCH --nodes 1
    #SBATCH -J justice181121
    #SBATCH --mail-user=aimalz@nyu.edu
    #SBATCH --mail-type=ALL
    #SBATCH -t 00:05:00
    #SBATCH --ntasks-per-node 28
    #run the application:
    bash {ROOT}/e2e-{}.sh
    '''


SH_TEMPLATE = '''
    #!/bin/bash
    module load python3/intel/3.6.3
    cd /home/$USER/justice
    source /scratch/$USER/venv3/bin/activate
    python3 -m justice.test.compute.example_end_to_end_worker --lc_ids {}
    '''


def dispatch(job_id, pairs):
    arg = ';'.join([f'{pair[0]},{pair[1]}' for pair in pairs])
    with open(f'{ROOT}/e2e-{job_id}.sl', 'w') as f:
        f.write(SL_TEMPLATE.format(job_id))
    with open(f'{ROOT}/e2e-{job_id}.sh', 'w') as f:
        f.write(SH_TEMPLATE.format(arg))
    subprocess.call(['sbatch', f'{ROOT}/e2e-{job_id}.sl'])


def gen_tasks():
    lc_ids = list(range(7, 100))
    tasks = ['{},{}'.format(a, b) for a in lc_ids for b in lc_ids if a != b]
    random.shuffle(tasks)
    for task in tasks:
        yield task


def main():
    queue = []
    for t in gen_tasks():
        queue.append(t)
        if len(queue) > 500:
            dispatch(queue)
            queue = []
    if queue:
        dispatch(queue)
    results_writer.scan_and_aggregate(
        pathlib.Path('/tmp/end-to-end-test'),
        'e2e')


if __name__ == '__main__':
    main()
