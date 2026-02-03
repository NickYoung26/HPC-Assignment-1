#!/opt/software/anaconda/python-3.10.9/bin/python

from mpi4py import MPI  # pylint: disable=import-error
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

N = 100_000_000
DELTA = 1.0 / N

CHUNK = 1_000_000  #Can likely be increased.

I_LOCAL = 0.0
step = nproc

for base in range(rank, N, step * CHUNK):
    # how many indices remain for this rank starting at 'base'
    remaining = (N - base + step - 1) // step   # ceil((N-base)/step)
    m = CHUNK if remaining > CHUNK else remaining

    k = np.arange(m, dtype=np.float64)
    x = (base + step * k + 0.5) * DELTA

    I_LOCAL += (4.0 / (1.0 + x * x)).sum()

I_LOCAL *= DELTA
I_FINAL = comm.reduce(I_LOCAL, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Good Code Integral {I_FINAL:.10f}")

