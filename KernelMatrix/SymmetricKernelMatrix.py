"""
Various implementations of symmetric kernel matrix calculations with different complexity trade-offs
"""
import multiprocessing, numpy, time

def _kernel_one_to_many(x, pts, kernel, kwargs, idx):
    xi = numpy.repeat(x.reshape((1,-1)), len(pts), axis=0)
    return idx, kernel(xi, pts, **kwargs)

class SymmetricMatrix:
    Q = None

    def __init__(self, X, kernel, kwargs, thread_pool = None, dtype = numpy.float16, verbose = True):
        self.verbose = verbose
        t0 = time.time()
        if X.dtype != dtype:
            X = X.astype(dtype)
        self.Q = self._symmetric_kernel_matrix(X, kernel, numpy.float16, thread_pool, kwargs)
        if self.verbose:
            print("Q matrix in", time.time()-t0)

    def _symmetric_kernel_matrix(self, X, kernel, result_dtype, thread_pool, kwargs):
        if thread_pool is not None:
            results = thread_pool.starmap(_kernel_one_to_many, [(X[i], X[i:], kernel, kwargs, i) for i in range(len(X))])
            # init Q after multiprocessing is done to preserve memory
            Q = numpy.zeros((len(X),len(X)), result_dtype)
            for idx, result in results:
                Q[idx,idx:] = Q[idx:,idx] = result
        else:
            Q = numpy.zeros((len(X),len(X)), result_dtype)
            for i in range(len(X)):
                Q[i,i:] = Q[i:,i] = _kernel_one_to_many(X[i], X[i:], kernel, kwargs, i)[1]
        return Q

    def matrix(self):
        return self.Q

    def __getitem__(self,idx):
        return self.Q[idx]

    def __imul__(self,m):
        self.Q *=  m
        return self

    def shrink(self, idx):
        self.Q = self.Q[idx,:][:,idx]

class SymmetricTriMatrix(SymmetricMatrix):
    pass

class SymmetricDynamicMatrix(SymmetricMatrix):
    pass
