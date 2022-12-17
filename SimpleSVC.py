# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:40:56 2016

@author: josiahw
"""
import numpy, time, numpy.linalg
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

def _emd(a,b):
    d = cdist(a, b)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / len(a)

def emdKernel(a,b, dim = 1):
    """
    N-dimensional earth-movers distance kernel
    """
    if len(a.shape) == dim+1:
        return numpy.array([_emd(a[i,:], b[i,:]) for i in range(a.shape[0])])
    return _emd(a,b)

def polyKernel(a,b,pwr):
    return numpy.dot(a,b.T)**pwr

def rbfKernel(a,b,gamma):
    """
    1-dimensional rbf kernel
    """
    #if len(a.shape) == 2:
    #    return numpy.exp(-gamma * numpy.linalg.norm(a - b, axis=-1))
    return numpy.exp(-gamma * numpy.linalg.norm(a - b, axis=-1))

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

    def _checkClass(self, a, b, steps):
        """
        This does a straight line interpolation between a and b, using n_checks number of segments.
        It returns True if a and b are connected by a high probability region, false otherwise.
        NOTE: authors originally suggested 20 segments but that is SLOOOOOW, so we use 3. In practice it is pretty good.
        """
        evals = numpy.array([self._predict_density(s * a + (1-s) * b) for s in steps])
        return numpy.amax(evals, axis=0)

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
                in_cluster = unvisited[checkVals <= self.b]
                if len(in_cluster) > 0:
                    c.extend(in_cluster)
                    unvisited = unvisited[checkVals > self.b]
                i += 1
            clusters.append(c)
            if self.verbose:
                print(f"Clustered {len(X)-len(unvisited)}/{len(X)} in {time.time()-t0}")

        #3: group components by classification
        self.classifications = numpy.zeros(len(X))
        for i in range(len(clusters)):
            for c in clusters[i]:
                self.classifications[c] = i
        if self.verbose:
            print(f"Clusters: {len(numpy.unique(self.classifications))}")
            print(f"Cluster sizes: {[len(c) for c in clusters]}")
        return self.classifications

    def fit_incremental(self, X, chunk_size = 2000):
        if X.shape[0] <= chunk_size:
            self.fit(X)
            return
        self.incremental = True
        self.fit(X[:int(chunk_size)])
        for i in range(1,int(numpy.ceil(X.shape[0] / chunk_size))):
            if (i+1)*chunk_size < X.shape[0]:
                self.fit(numpy.concatenate([self.sv, X[int(i*chunk_size):int((i+1)*chunk_size)]]))
            else:
                self.incremental = False
                self.fit(numpy.concatenate([self.sv, X[int(i*chunk_size):X.shape[0]]]))

    def fit(self, X):
        """
        Fit to data X with labels y.
        """

        """
        Construct the Q matrix for solving
        """
        Q = numpy.zeros((len(X),len(X)), dtype = self.dtype)
        if self.incremental and self.Qshrunk is not None:
            Q[:self.Qshrunk.shape[0],:self.Qshrunk.shape[1]] = self.Qshrunk
        for i in range(len(X)):
            start = i
            if self.incremental and self.Qshrunk is not None:
                start = self.Qshrunk.shape[1]
            Q[i,start:] = self.kernel(numpy.tile(X[i], (X.shape[0]-start, 1)), X[start:], **self.kwargs) 

        """
        Solve for a and w simultaneously by coordinate descent.
        This means no quadratic solver is needed!
        The support vectors correspond to non-zero values in a.
        """
        self.w = numpy.zeros(X.shape[1])
        self.a = numpy.zeros(X.shape[0])
        delta = 10000000000.0
        maxDelta = delta
        # X_range keeps a record of values with non-zero alphas so we can reduce compute as we converge
        X_range = range(len(X))[:]
        while delta > self.tolerance:
            delta = 0.
            for i in X_range:
                g = numpy.dot(Q[i], self.a) / Q[i,i] - 1.0
                adelta = self.a[i] - min(max(self.a[i] - g, 0.0), self.C)
                self.w += adelta * X[i]
                delta += abs(adelta)
                self.a[i] -= adelta
            delta /= len(X_range)
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
        self.sv = X[self.a >= self.C/100., :]
        Qshrunk = Q[self.a >= self.C/100.,:][:,self.a >= self.C/100.]
        if self.incremental:
            # TODO: figure out what to save for incrementals
            self.Qshrunk = Qshrunk
            return
        self.a = (self.a)[self.a >= self.C/100.]

        #calculate the contribution of all SVs
        Qshrunk *= numpy.dot(self.a.reshape((-1,1)),self.a.reshape((1,-1)))


        #this is needed for radius calculation apparently
        self.bOffset = numpy.sum(Qshrunk.ravel())
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
        if (len(X.shape) < self.dims+1):
            X = X.reshape((1,-1))
        clss = self.kernel(X, X, **self.kwargs)
        for i in range(X.shape[0]):
            clss[i] -= 2 * numpy.sum(self.a * self.kernel(X[i,:], self.sv, **self.kwargs))

        return (clss+self.bOffset)**0.5

    def _predict(self, X):
        """
        Predict class for a single sample of data
        """
        step_size = 1.0/(self.class_check_steps+1)
        steps = numpy.arange(step_size,1.0-step_size,step_size, dtype=X.dtype)
        for i in range(self.sv.shape[0]):
            score = self._checkClass(X,self.sv[i,:], steps)
            if score < self.b:
                return self.classifications[i]
        return -1

    def predict(self, X):
        """
        Predict classes for out of sample data X
        """        
        if len(X.shape) > self.dims:
            return numpy.array([self._predict(X[i,:]) for i in range(X.shape[0])])
        return self._predict(X)


if __name__ == '__main__':
    import sklearn.datasets, time, matplotlib
    data,labels = sklearn.datasets.make_moons(10000,noise=0.05,random_state=0)
    data -= numpy.mean(data,axis=0)

    #parameters can be sensitive, these ones work for two moons
    C = 0.1
    clss = SimpleSVClustering(C,1e-8,numpy.float32,rbfKernel,gamma=12.5)
    t0 = time.time()
    clss.fit_incremental(data)
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
