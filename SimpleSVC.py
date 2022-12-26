# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:40:56 2016

@author: josiahw
"""
import numpy, time, numpy.linalg
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import chi2_kernel, rbf_kernel, polynomial_kernel
from MultiThreadedExecutor import MultiThreadedExecutor
# we can have div by 0 in gradient descent
numpy.seterr(divide='ignore')

class SimpleSVClustering:
    w = None
    a = None
    b = None
    C = None
    sv = None
    kernel = None
    tolerance = None
    verbose = True
    dims = 1
    class_check_steps = 4
    incremental = False
    Qshrunk = None
    nystroem = None

    def __init__(self,
                 C,
                 tolerance = 0.001,
                 dtype = numpy.float32,
                 kernel = numpy.dot,
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
        self.dtype = dtype
        self.kwargs = kwargs
        self.k_neighbours = 60
        self.threadDispatcher = MultiThreadedExecutor()

        # set up interpolation step array (for checking connectedness)
        step_size = 1.0/(self.class_check_steps+1)
        steps = numpy.arange(step_size,1.0-step_size,step_size, dtype=self.dtype)
        # order steps from the middle out, as that's the most likely spot to violate connectedness
        middle_out = list(range(len(steps)))
        middle_out.sort(key = lambda x: abs(len(steps) - x))
        steps = steps[middle_out]
        self.data_interpolation_steps = steps

    def _checkClass(self, a, b, steps):
        """
        This does a straight line interpolation between a and b, using n_checks number of segments.
        It returns True if a and b are connected by a high probability region, false otherwise.
        NOTE: authors originally suggested 20 segments but that is SLOOOOOW, so we use 4. In practice it is pretty good.
        """
        evals = None
        #make sure b is the longer set of values
        if len(a.shape) > 1:
            c = b
            b = a
            a = c
        if len(b.shape) > 1:
            # record all values that violate connectedness assumptions, and only re-test those that don't yet violate
            evals = self._predict_density(steps[0] * a + (1-steps[0]) * b)
            candidates = numpy.unravel_index(numpy.flatnonzero(evals < self.b),evals.shape)[0]
            for s in steps[1:]:
                tests = self._predict_density(s * a + (1-s) * b[candidates])
                evals[candidates] = tests
                candidates = numpy.unravel_index(numpy.flatnonzero(evals < self.b),evals.shape)[0]
                if len(candidates) == 0:
                    break
        else: 
            # single-point case: return as soon as connectedness assumption is violated
            for s in steps:
                evals = [self._predict_density(s * a + (1-s) * b)]
                if evals[0] >= self.b:
                    break
        return evals

    def _getAllClasses(self, X, Q):
        """
        Assign class labels to each vector based on connected graph components.
        
        """
        # Warning: we assume that only support vectors are fed into this function

        # 1: build the connected clusters
        # Visit unvisited points in order of largest alpha first, as it's likely to have direct contact with the most in-class points
        # TODO: could be faster if refactored to use sets?.
        unvisited = numpy.array(list(range(len(X))))[numpy.argsort(self.a)[::-1]]
        clusters = []
        steps = self.data_interpolation_steps
        # test only k nearest neighbours - any further points will likely be connected to one of the knn's anyway
        fit_qValue = numpy.quantile(Q.ravel(),1-min(self.k_neighbours, len(X))/len(X))
        while len(unvisited):
            # create a new cluster with the first unvisited node
            c = [unvisited[0]]
            unvisited = unvisited[1:]
            i = 0
            t0 = time.time()
            while i < len(c) and len(unvisited):
                # for all nodes in the cluster, add all connected unvisited nodes and remove them from the unvisited list
                candidates = unvisited[Q[c[i], unvisited].ravel() > fit_qValue].ravel()
                checkVals = self._checkClass(X[c[i]], X[candidates], steps)
                in_cluster = candidates[checkVals <= self.b]
                if len(in_cluster) > 0:
                    noncandidates = unvisited[Q[c[i], unvisited].ravel() <= fit_qValue].astype(numpy.int32)
                    c.extend(in_cluster)
                    unvisited = numpy.concatenate([candidates[checkVals > self.b], noncandidates])
                i += 1
            clusters.append(c)
            
            if self.verbose:
                print(f"Clustered {len(X)-len(unvisited)}/{len(X)} in {time.time()-t0}")
        # sort so largest clusters have the lowest index - allows us to easily remove outlier clusters later
        clusters.sort(key=len, reverse=True)

        # 3: group components by classification
        self.classifications = numpy.zeros(len(X))
        for i in range(len(clusters)):
            for c in clusters[i]:
                self.classifications[c] = i
        if self.verbose:
            print(f"Clusters: {len(numpy.unique(self.classifications))}")
            print(f"Cluster sizes: {[len(c) for c in clusters]}")

    def fit_incremental(self, X, chunk_size = 5000):
        """
        fit data for SVM, using chunk_size points at a time and keeping previous support vectors
        """
        # TODO: rewrite to be more efficient by merging chunked solutions (i.e. binary heap merges)
        t0 = time.time()
        if X.shape[0] <= chunk_size:
            self.incremental = False
            self.fit(X)
            if self.verbose:
                print(f"{len(self.sv)} support vectors fit in {time.time()-t0}s")
            return
        self.incremental = True
        Q, sv = self.fit(X[:int(chunk_size)])
        chunk_stack = [(1, Q, sv)]
        if self.verbose:
                print(f"{len(self.sv)} support vectors fit in {time.time()-t0}s")
        i = 1
        # merge chunks of equal depth in a binary merge strategy to ensure the least number of merges to finish the job
        while i < int(numpy.ceil(X.shape[0] / chunk_size)) or len(chunk_stack) > 1:
            # enqueue a new chunk of points
            if len(chunk_stack) < 2 or chunk_stack[-1][0] != chunk_stack[-2][0] and i < int(numpy.ceil(X.shape[0] / chunk_size)):
                start_index = int(i*chunk_size)
                end_index = int(min((i+1)*chunk_size, X.shape[0]))
                Q, sv = self.fit(numpy.concatenate([self.sv, X[start_index:end_index]]), Q)
                chunk_stack.append((1, Q, sv))
                i = i + 1
            # merge existing chunks of points
            elif len(chunk_stack) > 1:
                chunk_depth, Q1, sv1 = chunk_stack.pop(-1)
                _, Q2, sv2 = chunk_stack.pop(-1)
                if self.verbose:
                    print(f"merging", len(sv1)+len(sv2), "points at depth", chunk_depth)
                # set incremental to false on the final merge
                if i >= int(numpy.ceil(X.shape[0] / chunk_size)) and len(chunk_stack) == 0: self.incremental = False
                Q, sv = self.fit(numpy.concatenate([sv1, sv2]), Q1, Q2)
                del sv1, sv2, Q1, Q2
                chunk_stack.append((chunk_depth + 1, Q, sv))
            if self.verbose:
                print(f"{len(self.sv)} support vectors fit in {time.time()-t0}s")

    def fit(self, X, Q1 = None, Q2 = None):
        """
        Fit to data X with labels y.
        """

        """
        Construct the Q matrix for solving
        """
        # TODO: implement nystrom approximation for Q to improve memory / compute
        min_start = 0
        max_end = len(X)
        if Q1 is None:
            Q = numpy.zeros((len(X),len(X)), dtype = self.dtype)
        else:
            Q = numpy.resize(Q1+1, (len(X),len(X)))
            min_start = Q1.shape[1]
            if not Q2 is None:
                Q[min_start:,min_start:] = Q2
                max_end = min_start
            del Q1, Q2
        for i in range(max_end):
            start = max(i, min_start)
            Q[start:, i] = Q[i, start:] = self.kernel(X[i].reshape(1, -1), X[start:], **self.kwargs)

        """
        Solve for a and w simultaneously by coordinate descent.
        This means no quadratic solver is needed!
        The support vectors correspond to non-zero values in a.
        """
        self.w = numpy.zeros(X.shape[1])
        self.a = numpy.zeros(X.shape[0])
        min_a_val = self.C/100.
        delta = 10000000000.0
        maxDelta = delta
        # X_range keeps a record of values with non-zero alphas so we can reduce compute as we converge
        X_range = range(len(X))[:]
        while delta > self.tolerance:
            delta = 0.
            g = None
            for i in X_range:
                g = numpy.nan_to_num(numpy.divide(numpy.dot(Q[i], self.a), Q[i,i])) - 1.0
                adelta = self.a[i] - min(max(self.a[i] - g, 0.0), self.C)
                self.w += adelta * X[i]
                delta += abs(adelta)
                self.a[i] -= adelta
            delta /= len(X_range)
            if delta < maxDelta/2 and maxDelta < 10000000000.0: # every time delta halves, remove points we are reasonably sure won't be SVs
                # stackoverflow says flatnonzero is faster than numpy.where()
                X_range = numpy.unravel_index(numpy.flatnonzero(self.a >= min_a_val),self.a.shape)[0]
                self.a *= self.a >= min_a_val
                maxDelta = delta
                if self.verbose:
                    print ("Descent step magnitude:", delta)
            elif maxDelta == 10000000000.0:
                maxDelta = delta
                if self.verbose:
                    print ("Descent step magnitude:", delta)

        # get the data for support vectors
        self.sv = X[self.a >= min_a_val, :]
        Qshrunk = Q[self.a >= min_a_val,:][:,self.a >= min_a_val]
        # stop early if we still ahve points to process
        if self.incremental:
            return Qshrunk, self.sv
        self.a = (self.a)[self.a >= min_a_val]

        # this is needed for radius calculation
        self.bOffset = numpy.sum((Qshrunk * numpy.dot(self.a.reshape((-1,1)),self.a.reshape((1,-1)))).ravel()) 
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
        self._getAllClasses(self.sv, Qshrunk)
        if self.verbose:
            print(f"Clusters assigned in {time.time()-t0}s")
        
        return Qshrunk, self.sv

    def _predict_density(self, X):
        """
        For SVClustering, we need to calculate radius rather than bias.
        """
        # multithreading speeds up 
        if len(X.shape) == 1:
            return numpy.sqrt(numpy.subtract(self.kernel(X.reshape(1, -1), X.reshape(1, -1), **self.kwargs) + self.bOffset, 2 * numpy.dot(self.a, self.kernel(X.reshape(1, -1), self.sv, **self.kwargs).ravel())))
        data_func = lambda x: numpy.subtract(self.kernel(x.reshape(1, -1), x.reshape(1, -1), **self.kwargs) + self.bOffset, 2 * numpy.dot(self.a, self.kernel(x.reshape(1, -1), self.sv, **self.kwargs).ravel()))
        return numpy.sqrt(self.threadDispatcher.fill(data_func, X))

    def predict(self, X):
        """
        Predict classes for out of sample data X
        """
        steps = self.data_interpolation_steps
        if len(X.shape) < 2:
            X = [X]
        classes = numpy.zeros(len(X), dtype=numpy.int64)-1
        for j in range(len(X)):
            for i in range(len(self.sv)):
                vals = self._checkClass(X[j], self.sv[i], steps)
                if vals[0] < self.b:
                    classes[j] = self.classifications[i]
                    break
        return classes


if __name__ == '__main__':
    import sklearn.datasets, time, matplotlib
    data,labels = sklearn.datasets.make_moons(10000,noise=0.05,random_state=0)
    data -= numpy.mean(data,axis=0)

    #parameters can be sensitive, these ones work for two moons
    C = 0.05
    clss = SimpleSVClustering(C,1e-7,numpy.float16,rbf_kernel,gamma=13)
    t0 = time.time()
    clss.fit_incremental(data,5000)
    print(f"fit in {time.time()-t0} seconds")

    #check assigned classes for the two moons as a classification error
    t0 = time.time()
    t = clss.predict(data)
    print(f"predicted in {time.time()-t0} seconds")
    print ("Error", numpy.sum((labels-t)**2) / float(len(data)))


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
