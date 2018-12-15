from justice.compute import results_writer
import random
import subprocess

ROOT = '/home/aim267/justice'


SL_TEMPLATE = '''
    #!/bin/bash -l
    #SBATCH --nodes 1
    #SBATCH -J justice181121
    #SBATCH --mail-user=aimalz@nyu.edu
    #SBATCH --mail-type=ALL
    #SBATCH -t 00:05:00
    #SBATCH --ntasks-per-node 28
    #run the application:
    bash {}/{}-{}.sh
    '''

SH_TEMPLATE = '''
    #!/bin/bash
    module load python3/intel/3.6.3
    cd /home/$USER/justice
    source /scratch/$USER/venv3/bin/activate
    python3 -m {} {}
    '''


class SlurmDispatcher:
    def __init__(self, module):
        self.module = module

    def _sl_script(self, job_id):
        return SL_TEMPLATE.format(ROOT, self.module, job_id)

    def _sh_script(self, params):
        words = []
        for k, v in params.items():
            words.extend([str(k), str(v)])
        kwarg_str = ' '.join(words)
        return SH_TEMPLATE.format(self.module, kwarg_str)

    def dispatch(self, job_id, params):
        with open(f'{ROOT}/{self.module}-{job_id}.sl', 'w') as f:
            f.write(self._sl_script(job_id))
        with open(f'{ROOT}/{self.module}-{job_id}.sh', 'w') as f:
            f.write(self._sh_script(params))
        subprocess.call(['sbatch', f'{ROOT}/{self.module}-{job_id}.sl'])


def main():
    params_list = [
        {},
        {'--window-size': 8},
        {'--window-size': 11, '--learning-rate': 0.002}
    ]
    dispatcher = SlurmDispatcher('justice.align_model.basic_train')
    for job_id, params in enumerate(params_list):
        dispatcher.dispatch(job_id, params)

if __name__ == '__main__':
    main()
