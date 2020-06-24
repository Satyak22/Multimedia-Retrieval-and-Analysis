from project_utils import *
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA

def visualize_latent_semantics(latent_semantic, image_ids, imageSpace=False):
    for i, u_0 in enumerate(latent_semantic):
        print("\nSemantic - ", i)
        temp = []
        if imageSpace:
            for j, elem in zip(image_ids, u_0):
                temp.append((j, elem))
        else:
            for j, elem in enumerate(u_0):
                temp.append((j, elem))
        print("Term Weight Pairs are:\n")
        for termWeightPair in sorted(temp, key=lambda tup: tup[1], reverse=True):
            print(termWeightPair)

def do_svd(data_mat, k):
    u, s, vt = np.linalg.svd(data_mat, full_matrices=True)
    u = u[:, :k]
    s = s[:k]
    vt = vt[:k, :]
    return u, s, vt

def reduce_dimensions_svd(data_mat, k, image_ids, viz=False, get_v=False):
    u, s, vt = do_svd(data_mat, k)
    d_reduced = np.matmul(u, np.diag(s))
    if viz:
        print("Reduced with SVD, new data shape is " , d_reduced.shape)
        print("Latent Semantics in Image Space")
        visualize_latent_semantics(u.T, image_ids, imageSpace=True)
        print("Latent Semantics in Feature Space")
        visualize_latent_semantics(vt, image_ids)
    if get_v:
        return vt.T
    else:
        return d_reduced

def reduce_dimensions_pca(data_mat, k, image_ids, viz=False, get_v=False):
    pca = PCA(n_components=k)
    d_reduced = pca.fit_transform(data_mat)
    if viz:
        print("Reduced with PCA, new data shape is ", d_reduced.shape)
        print("Latent Semantics Image Space")
        visualize_latent_semantics(d_reduced.T, image_ids, imageSpace=True)
        #print("Latent Semantics Feature Space")
        visualize_latent_semantics(pca.components_,image_ids)
    if get_v:
        return pca.components_.T
    else:
        return d_reduced

def reduce_dimensions_lda(data_mat, k, image_ids, viz=False, get_v=False):
    lda = LatentDirichletAllocation(n_components=k, random_state = 0)
    d_reduced = lda.fit_transform(data_mat)
    if viz:
        print("Reduced with LDA, new data shape is ", d_reduced.shape)
        print("The latent semantics in image space is")
        visualize_latent_semantics(d_reduced.T, image_ids, imageSpace=True)
        print("Latent Semantics in feature space are")
        visualize_latent_semantics(lda.components_, image_ids)
    if get_v:
        return lda.components_
    else:
        return d_reduced

def reduce_dimensions_nmf(data_mat, k, image_ids, viz=False, get_v=False):
    nmf = NMF(n_components=k, init='random', random_state=0)
    d_reduced = nmf.fit_transform(data_mat)
    if viz:
        print("\n######## Reduced with NMF, new data shape is ", d_reduced.shape)
        print("\n######## The latent semantics in image space is #######")
        visualize_latent_semantics(d_reduced.T, image_ids, imageSpace=True)
        print("\n######## The latent semantics in feature space is #######")
        visualize_latent_semantics(nmf.components_, image_ids)
    if get_v:
        return nmf.components_
    else:
        return d_reduced
