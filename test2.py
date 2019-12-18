import numpy as np
from mpi4py import MPI
import time

def square(x):
    sq = 0
    for y in x:
        sq += y ** 2
    return sq


def main():
    np.random.seed(2)
    seqLen = 10000000
    x = np.random.randn(seqLen)
    start = time.time()
    # block for simple program
    res = square(x)
    # block for MPI
    # this block running with following command:
    # mpiexec -n 4 python mpiProg.py
    # 4 - number of nodes
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    i = len(x) // size
    buf = x[i * rank: i * (rank + 1)]
    sq = square(buf)
    res = comm.allreduce(sq)
    '''
    end = time.time()
    print(end - start)
    procArr = [1, 2, 4, 8]
    timeArr = [5.02, 3.22, 2.92, 2.9]
    return 0


if __name__ == '__main__':
    main()