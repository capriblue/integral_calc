import subprocess

import numpy as np

nitn = 20  # あんまりいじっても意味ない 10~20が最適らしい
neval = 1e+8  # 精度が欲しい場合はここを大きくする1/sqrt(neval)くらいに精度が上がる

qs = np.array([1])
phi_0s = np.array([0.])
k_FLs = np.array([10,])
kF_lambdas = np.linspace(0.01, 2, 10)

# if you does not need the output, set bellow to /dev/null
output = "/dev/null"  # "slurm-%j.out"  #
path_prefix = "result_FOMP"
job_name = "JOBNAME"

for q in qs:
    for phi_0 in phi_0s:
        for k_FL in k_FLs:
            for kF_lambda in kF_lambdas:
                subprocess.run(
                    f"sbatch -J {job_name} -N 1 -c 1 -p FOMP --output={output} --wrap='python integral.py {kF_lambda} --q {q} --phi_0 {phi_0} --k_FL {k_FL} --nitn {nitn} --neval {neval} --path {path_prefix}'", shell=True)
