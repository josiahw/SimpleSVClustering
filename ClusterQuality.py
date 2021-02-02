"""
Cluster evaluation algo from
"Clustering Evaluation In Feature Space"
"""
import numpy

def _calcMatrix(pts1, pts2, kernel, thread_pool = None, **kwargs):
    result = numpy.zeros((pts1.shape[0], pts2.shape[0]))
    for i in range(pts1.shape[0]):
        for j in range(i,pts2.shape[0]):
            result[i,j] = result[j,i] = kernel(pts1[i],pts2[j],**kwargs)
    return result

def KDB(pts, membership, kernel, **kwargs):
    """
    Kernel Davies-Bouldin index - returns a value measuring quality of clusters
    """
    # 1 get cluster matrices
    cluster_ids = numpy.unique(membership)
    clusters = [pts[id == membership,:] for id in cluster_ids]
    cluster_matrices = [_calcMatrix(p,p,kernel,**kwargs) for p in clusters]

    # 2 calculate R for every cluster pair
    R_vals = []
    for i in range(cluster_ids.shape[0]):
        for j in range(cluster_ids.shape[0]):
            Cij = _calcMatrix(clusters[i],clusters[j],kernel,**kwargs)
            R_vals.append(cluster_matrices[i], cluster_matrices[j], Cij)

    # 3 return R index
    return sum(R_vals)/len(R_vals)

def R(cluster1, cluster2, cluster12):
    """
    2-cluster measure of spread ratios
    """
    result = (Cj(cluster1) + Cj(cluster2)) / dCjCm(cluster1, cluster2, cluster12)

def Cj(cluster):
    """
    Within-cluster measure of spread
    """
    # do de kernel thing
    result = 1.0/cluster.shape[0] * numyp.sum(numpy.diag(cluster)) - 2.0 / cluster.shape[0]**2 * numpy.sum(cluster.ravel())
    return result

def dCjCm(cluster1, cluster2, cluster12):
    """
    Intra-cluster measure of spread
    """
    result = 1.0 / (cluster1.shape[0] * cluster1.shape[1]) * numpy.sum(cluster1.ravel())
    result += 1.0 / (cluster2.shape[0] * cluster2.shape[1]) * numpy.sum(cluster2.ravel())
    return result
