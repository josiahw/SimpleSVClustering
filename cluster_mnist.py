"""

"""
import numpy
from sklearn.datasets import fetch_openml
from SimpleSVC import SimpleSVClustering, emdRbfKernel
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1)).astype(numpy.float16)

# 60k samples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=3000, test_size=20000)
del X, y
import sklearn.datasets, time, matplotlib

#parameters can be sensitive, these ones work for two moons
C = 0.05
gamma = numpy.array([0.012], dtype=numpy.float16)
clss = SimpleSVClustering(C, 1e-8, emdRbfKernel, gamma=gamma, dim=(28,28))
t0 = time.time()
clss.fit(X_train)
print(f"fit in {time.time()-t0} seconds")

#check assigned classes for the two moons as a classification error
t0 = time.time()
t = clss.predict(X_test)
print(f"predicted in {time.time()-t0} seconds")

clusters = numpy.unique(t[t != -1])
print(f"{len(clusters)} classes detected")
classes = y_test.unique()
cluster_class = {}
correct = 0.0
for cls in classes:
    scores = numpy.zeros(len(clusters))
    for r in t[y_test == cls]:
        scores[r] += 1
    cluster_class[cls] = numpy.argmax(scores)
    correct += numpy.max(scores)
print(f"Accuracy: {correct / len(t)}")

t0 = time.time()
from ClusterQuality import KDB
print ("KDB", KDB(X_train, t, clss.kernel, **clss.kwargs))
print(f"KDB calculated in {time.time()-t0}")
