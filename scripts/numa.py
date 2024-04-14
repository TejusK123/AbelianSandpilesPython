from numba import numba, cuda, guvectorize, int64
import numpy as np




from numba import guvectorize, int64, float64

@guvectorize([(int64[:, :], int64[:, :])], '(n),()->(n)')
def g(x, res):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > 3:
                res[i][j] = x[i][j] - 4
            # if x[i+1][j] > 3:
            #     res[i][j] += 1
            # if x[i][j+1] > 3:
            #     res[i][j] += 1
            # if x[i-1][j] > 3:
            #     res[i][j] += 1
            # if x[i][j-1] > 3:
            #     res[i][j] += 1



test1 = np.full((10,10), 4, dtype = np.int64)

print(g(test1))