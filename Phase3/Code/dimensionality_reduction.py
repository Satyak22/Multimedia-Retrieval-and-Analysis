import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA

def do_svd(data_mat, k):
    u, s, vt = np.linalg.svd(data_mat, full_matrices=True)
    u = u[:, :k]
    s = s[:k]
    vt = vt[:k, :]
    return u, s, vt

def reduce_dimensions_svd(data_mat, k, get_v=False):
    print("reducing dimensions...")
    u, s, vt = do_svd(data_mat, k)
    d_reduced = np.matmul(u, np.diag(s))
    if get_v:
        return d_reduced, vt.T
    else:
        return d_reduced

def reduce_dimensions_pca(data_mat, k, get_v=False):
    pca = PCA(n_components=k)
    d_reduced = pca.fit_transform(data_mat)
    if get_v:
        return d_reduced, pca.components_.T
    else:
        return d_reduced

def reduce_dimensions_lda(data_mat, k, get_v=False):
    print("reducing dimensions...")
    lda = LatentDirichletAllocation(n_components=k, random_state = 0)
    d_reduced = lda.fit_transform(data_mat)
    if get_v:
        return d_reduced, lda.components_.T
    else:
        return d_reduced

def full_reduce_dimensions_lda(data_mat, k, get_v=False):
    print("reducing dimensions...")
    lda = LatentDirichletAllocation(n_components=k, random_state = 0)
    d_reduced = lda.fit_transform(data_mat)
    if get_v:
        return d_reduced, lda.components_.T
    else:
        return d_reduced


def reduce_dimensions_nmf(data_mat, k, get_v=False):
    nmf = NMF(n_components=k, init='random', random_state=0)
    d_reduced = nmf.fit_transform(data_mat)
    if get_v:
        return d_reduced, nmf.components_
    else:
        return d_reduced
