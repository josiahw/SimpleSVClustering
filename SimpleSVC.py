# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:40:56 2016

@author: josiahw
"""
import numpy, time, numpy.linalg, os

# these are not essential, they are specifically for earth movers distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy import stats
from multiprocessing import Pool, cpu_count
from pyemd import emd
_svc_pts = None
from KernelMatrix.SymmetricKernelMatrix import SymmetricMatrix

def _kernel_one_to_global(x, kernel, kwargs, idx):
    xi = numpy.repeat(x.reshape((1,-1)), len(_svc_pts), axis=0)
    return idx, kernel(xi, _svc_pts, **kwargs)

def _load_global_pts(pts):
    global _svc_pts
    _svc_pts = pts

def _kernel_matrix(pts1, pts2, kernel, result_dtype, thread_pool, kwargs):
    Q = numpy.zeros((len(pts1),len(pts2)), result_dtype)
    if thread_pool is not None:
        for idx, result in thread_pool.starmap(_kernel_one_to_many, [(pts1[i], pts2, kwargs, i) for i in range(len(pts1))]):
            Q[idx] = result
    else:
        for i in range(len(pts1)):
            Q[i] = _kernel_one_to_many(pts1[i], pts2, kernel, kwargs, i)[1]
    return Q

def _kernel_matrix_global(pts1, kernel, result_dtype, thread_pool, kwargs):
    """
    This requires a thread pool loaded with an init function to global pts
    """
    global _svc_pts
    Q = numpy.zeros((len(pts1),len(_svc_pts)), result_dtype)
    if thread_pool is not None:
        for idx, result in thread_pool.starmap(_kernel_one_to_global, [(pts1[i], kernel, kwargs, i) for i in range(len(pts1))]):
            Q[idx] = result
    else:
        for i in range(len(pts1)):
            Q[i] = _kernel_one_to_many(pts1[i], kernel, kwargs, i)[1]
    return Q

def _sumreduce_global(x, mul_val, kernel, kwargs, idx):
    return idx, numpy.sum(mul_val * _kernel_one_to_global(x, kernel, kwargs, idx)[1])

def _matrix_sumreduce_global(pts1, mul_val, kernel, result_dtype, thread_pool, kwargs):
    """
    multiply matrix by mul_val and sum to reduce
    """
    global _svc_pts
    Q = numpy.zeros((len(pts1),), result_dtype)
    if thread_pool is not None:
        for idx, result in thread_pool.starmap(_sumreduce_global, [(pts1[i], mul_val, kernel, kwargs, i) for i in range(len(pts1))]):
            Q[idx] = result
    else:
        for i in range(len(pts1)):
            Q[i] = _sumreduce_global(pts1[i], mul_val, kernel, kwargs, i)[1]
    return Q

_dist_mat = None
def emdRbfKernel2(p1, p2, dim, gamma=1.0):
    p1 = p1.ravel()
    p2= p2.ravel()
    global _dist_mat
    if _dist_mat is None:
        as_pt = lambda x: numpy.array([int(x/float(dim[0])), x%dim[0]], dtype=p1.dtype)
        print(as_pt(78) - as_pt(147))
        _dist_mat = numpy.array([[numpy.linalg.norm(as_pt(i) - as_pt(j)) for j in range(len(p2))] for i in range(len(p1))], dtype=p1.dtype)
    print(emd(p1, p2, _dist_mat))
    return emd(p1, p2, _dist_mat)

def _emd(a,b):
    d = cdist(a, b)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / len(a)

def emdRbfKernel(a,b, dim = (-1,), gamma = 1.0):
    """
    N-dimensional earth-movers distance kernel
    """
    if len(a.shape) == 2:
        return numpy.exp(-gamma * numpy.array([_emd(a[i].reshape(dim), b[i].reshape(dim)) for i in range(a.shape[0])]))
    return numpy.exp(-gamma * _emd(a.reshape(dim),b.reshape(dim)))

def polyKernel(a,b,pwr):
    return numpy.dot(a,b.T)**pwr

def rbfKernel(a,b,gamma):
    """
    1-dimensional rbf kernel
    """
    if len(a.shape) == 2:
        return numpy.exp(-gamma * numpy.linalg.norm(a - b, axis=1))
    return numpy.exp(-gamma * numpy.linalg.norm(a - b, axis=0))

class SimpleSVClustering:
    w = None
    a = None
    b = None
    C = None
    sv = None
    kernel = None
    tolerance = None
    thread_pool = Pool(6)
    verbose = True
    dims = 1
    class_check_steps = 3

    def __init__(self,
                 C,
                 tolerance = 0.001,
                 kernel = numpy.dot,
                 dtype = numpy.float16,
                 **kwargs
                 ):
        """
        The parameters are:
         - C: SVC cost
         - tolerance: gradient descent solution accuracy
         - kernel: the kernel function do use as k(a, b, *kwargs)
         - kwargs: extra parameters for the kernel
        """
        self.C = C
        self.kernel = kernel
        self.tolerance = tolerance
        self.kwargs = kwargs
        self.dtype =dtype
        step_size = 1.0/(self.class_check_steps+1)
        steps = numpy.arange(step_size,1.0-step_size,step_size, dtype=dtype)
        self.a_weights = steps.reshape(-1,1)
        self.b_weights = (1-steps).reshape(-1,1)

    def _checkClass(self, a, b, steps):
        """
        This does a straight line interpolation between a and b, using n_checks number of segments.
        It returns True if a and b are connected by a high probability region, false otherwise.
        NOTE: authors originally suggested 20 segments but that is SLOOOOOW, so we use 4. In practice it is pretty good.
        """
        checkVals = self._predict_density(steps[0] * a + (1-steps[0]) * b)
        for s in steps[1:]:
            checkVals = numpy.maximum(checkVals, self._predict_density(s * a + (1-s) * b))
        return checkVals

    def _getAllClasses(self, X):
        """
        Assign class labels to each vector based on connected graph components.
        TODO: The outputs of this should really be saved in order to embed new points into the clusters.
        """
        #1: build the connected clusters
        unvisited = numpy.array(list(range(len(X))))
        clusters = []
        step_size = 1.0/(self.class_check_steps+1)
        steps = numpy.arange(step_size,1.0-step_size,step_size, dtype=X.dtype)
        while len(unvisited):
            #create a new cluster with the first unvisited node
            c = [unvisited[0]]
            unvisited = unvisited[1:]
            i = 0
            t0 = time.time()
            while i < len(c) and len(unvisited):
                #for all nodes in the cluster, add all connected unvisited nodes and remove them fromt he unvisited list
                candidate_list = X[unvisited]
                checkVals = self._checkClass(X[c[i]], candidate_list, steps)
                in_cluster = numpy.where((checkVals <= self.b).ravel())[0]
                if len(in_cluster) > 0:
                    c.extend([unvisited[j] for j in in_cluster])
                    out_cluster = numpy.where((checkVals > self.b).ravel())[0]
                    unvisited = [unvisited[j] for j in out_cluster]
                i += 1
            clusters.append(c)
            if self.verbose:
                print(f"Clustered {len(X)-len(unvisited)}/{len(X)} in {time.time()-t0}")

        #3: group components by classification
        self.classifications = numpy.zeros(len(X))
        for i in range(len(clusters)):
            for c in clusters[i]:
                self.classifications[c] = i
        print(f"Clusters: {len(numpy.unique(self.classifications))}")
        print(f"Cluster sizes: {[len(c) for c in clusters]}")
        return self.classifications

    def fit(self, X):
        """
        Fit to data X with labels y.
        """

        """
        Construct the Q matrix for solving
        """
        Q = SymmetricMatrix(X, self.kernel, self.kwargs, self.thread_pool, numpy.float16, self.verbose)

        """
        Solve for a and w simultaneously by coordinate descent.
        This means no quadratic solver is needed!
        The support vectors correspond to non-zero values in a.
        """
        self.w = numpy.zeros(X.shape[1])
        self.a = numpy.zeros(X.shape[0])
        delta = 10000000000.0
        maxDelta = delta
        X_range = range(len(X))[:]
        while delta > self.tolerance:
            delta = 0.
            for i in X_range:
                g = numpy.dot(Q[i], self.a) / Q[i,i] - 1.0
                adelta = self.a[i] - min(max(self.a[i] - g, 0.0), self.C)
                self.w += adelta * X[i]
                delta += abs(adelta)
                self.a[i] -= adelta
            delta /=  len(X_range)
            if delta < maxDelta/2 and maxDelta < 10000000000.0: # every time delta halves, remove points we are reasonably sure won't be SVs
                X_range = numpy.where(self.a >= self.C/1000.)[0]
                maxDelta = delta
                if self.verbose:
                    print ("Descent step magnitude:", delta)
            elif maxDelta == 10000000000.0:
                maxDelta = delta
                if self.verbose:
                    print ("Descent step magnitude:", delta)

        #get the data for support vectors
        Q.shrink(self.a >= self.C/100.)
        self.sv = X[self.a >= self.C/100., :]
        self.a = (self.a)[self.a >= self.C/100.]

        #calculate the contribution of all SVs
        Q *= numpy.dot(self.a.reshape((-1,1)),self.a.reshape((1,-1)))

        #this is needed for radius calculation
        self.bOffset = numpy.sum(Q.matrix().ravel())

        if self.verbose:
            print ("Number of support vectors:", len(self.a))

        """
        Select support vectors and solve for b to get the final classifier
        """
        t0 = time.time()
        self.b = numpy.mean(self._predict_density(self.sv))

        if self.verbose:
            print ("Bias value:", self.b, f"in {time.time()-t0}s")

        """
        Assign clusters to training dataset
        """
        t0 = time.time()
        self._getAllClasses(self.sv)
        if self.verbose:
            print(f"Clusters assigned in {time.time()-t0}s")

    def _predict_density(self, X):
        """
        For SVClustering, we need to calculate radius rather than bias.
        """
        if (len(X.shape) == self.dims):
            X = X.reshape((1,-1))
        clss = self.kernel(X, X, **self.kwargs)

        global _svc_pts
        if _svc_pts is None:
            num_processes = self.thread_pool._processes
            self.thread_pool.close()
            self.thread_pool = Pool(num_processes, initializer=_load_global_pts, initargs=(self.sv,))
            _load_global_pts(self.sv)
        results = _matrix_sumreduce_global(X, self.a, self.kernel, numpy.float16, self.thread_pool, self.kwargs)
        clss -= 2 * results
        return (clss+self.bOffset)**0.5

    def predict(self, X):
        """
        Predict classes for out of sample data X
        """
        step_size = 1.0/(self.class_check_steps+1)
        steps = numpy.arange(step_size,1.0-step_size,step_size, dtype=X.dtype)
        if len(X.shape) == 1:
            X = X.reshape((1,-1))
        results = numpy.zeros(len(X))-1
        for i in range(len(X)):
            checkVals = self._checkClass(X[i], self.sv, steps)
            in_cluster = numpy.where((checkVals <= self.b).ravel())[0]
            if len(in_cluster):
                #TODO: make this more resilient
                results[i] = self.classifications[in_cluster[0]] #stats.mode(self.classifications[in_cluster])
        return results


if __name__ == '__main__':
    import sklearn.datasets, time, matplotlib
    data,labels = sklearn.datasets.make_moons(5000,noise=0.05,random_state=0)
    data -= numpy.mean(data,axis=0)

    #parameters can be sensitive, these ones work for two moons
    C = 0.1
    gamma = numpy.array([12.5], dtype=numpy.float32)
    clss = SimpleSVClustering(C,1e-8,rbfKernel,gamma=gamma)
    t0 = time.time()
    clss.fit(data)
    print(f"fit in {time.time()-t0} seconds")

    #check assigned classes for the two moons as a classification error
    t0 = time.time()
    t = clss.predict(data)
    from ClusterQuality import KDB
    print(f"predicted in {time.time()-t0} seconds")
    print ("Error", numpy.sum((labels-t)**2) / float(len(data)))
    t0 = time.time()
    print ("KDB", KDB(data, t, clss.kernel, **clss.kwargs))
    print(f"KDB calculated in {time.time()-t0}")


    from matplotlib import pyplot

    #generate a heatmap and display classified clusters.
    matplotlib.use("Agg")
    a = numpy.zeros((100,100))
    for i in range(100):
        for j in range(100):
            a[j,i] = clss._predict_density(numpy.array([i*4/100.-2,j*4/100.-2]))
    pyplot.imshow(a, cmap='hot', interpolation='nearest')
    data *= 25.
    data += 50.
    pyplot.scatter(data[t==0,0],data[t==0,1],c='r')
    pyplot.scatter(data[t==1,0],data[t==1,1],c='b')
    pyplot.savefig("out.png")
    #pyplot.show()
