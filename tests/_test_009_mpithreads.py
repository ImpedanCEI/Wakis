# ruff: noqa: E402

import subprocess

sockets, cores = map(
    int,
    subprocess.check_output(
        "lscpu | awk -F: '/Socket\\(s\\)/{s=$2} /Core\\(s\\) per socket/{c=$2} END{print s,c}'",
        shell=True,
    ).split(),
)
print(f"Sockets={sockets}, Cores/Socket={cores}")

import os

# Set before importing numpy/scipy/mkl modules
os.environ["OMP_NUM_THREADS"] = str(cores)  # Number of OpenMP threads
os.environ["KMP_AFFINITY"] = "balanced,granularity=fine"


import numpy as np
from mpi4py import MPI
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack

try:
    from sparse_dot_mkl import csr_matrix, dot_product_mkl
except Exception as e:
    print("sparse_dot_mkl not available:", e)

import timeit

from threadpoolctl import threadpool_info

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

nt = 1 // p  # int(os.environ['SLURM_CPUS_PER_TASK']) # num threads per task
user_api = threadpool_info()[0]["user_api"]
# threadctl = threadpool_limits(limits=nt, user_api='blas')
# threadctl = threadpool_limits(limits=nt, user_api='openmp')
N = 80000000 // p  # Matrix size

# Build curl
Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
Py = diags([-1, 1], [0, 10], shape=(N, N), dtype=np.int8)
Pz = diags([-1, 1], [0, 100], shape=(N, N), dtype=np.int8)
A = vstack(
    [
        hstack([sparse_mat((N, N)), -Pz, Py]),
        hstack([Pz, sparse_mat((N, N)), -Px]),
        hstack([-Py, Px, sparse_mat((N, N))]),
    ],
    dtype=np.float64,
)

B = np.random.rand(3 * N)

try:
    print("Using MKL backend")
    A = csr_matrix(A)
except Exception as e:
    print("Using Scipy backend:", e)
    A = A.tocsr()

print(B.shape)


def test_matmul():
    # with threadpool_limits(limits=nt, user_api='blas'):
    # print(threadpool_info())
    # A = np.random.rand(N, N)
    # B = np.random.rand(1, N)
    # C = np.matmul(A, B)

    try:
        C = dot_product_mkl(A, B)
    except Exception as e:
        print("Falling back to Scipy dot:", e)
        C = A.dot(B)
    print(C.shape)


# if my_rank == 0:
n_runs = 10
t = timeit.timeit(
    "test_matmul()", setup="from __main__ import test_matmul", number=n_runs
)

print(f"Rank {my_rank}: Average time over {n_runs} runs: {t / n_runs:.4f} seconds")


MPI.Finalize()
