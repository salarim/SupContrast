import os
import subprocess
from itertools import product

# env_vars = [
#             first_dependent_set,
#             second_dependent_set,
#             ...,
#             ith_dependent_set={'first_var':[ordered_possible_values_for_the_first_var],
#                                   ...,
#                                 }
#             ]
env_vars = [{'BATCH_SIZE':['2048','1024','512','256','128'],
             'VIEWS':['2','4','8','16','30'],
             'DROP_RATIO':['0.0','0.5','0.75','0.875','0.933']}]

independent_sets = []
for independent_dict in env_vars:
    independent_set = []
    for values in zip(*independent_dict.values()):
        d = {}
        for i, k in enumerate(independent_dict.keys()):
            d[k] = values[i]
        independent_set.append(d)
    independent_sets.append(independent_set)

for env_dicts in product(*independent_sets):
    for env_dict in env_dicts:
        for k, v in env_dict.items():
            os.environ[k] = v
    job_name = "SimCLR-shapenet-bs{}-v{}-dr{}-e100-resnet18".format(os.environ['BATCH_SIZE'],
                                                                    os.environ['VIEWS'],
                                                                    os.environ['DROP_RATIO'])

    rc = subprocess.check_call(["sbatch",
                                "--job-name={}".format(job_name),
                                "job.sh"])

