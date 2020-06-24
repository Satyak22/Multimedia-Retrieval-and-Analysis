# Multimedia Retrieval and Analysis

The project aimed at various techniques for feature extraction, dimensionality reduction and how these techniques contribute to similarity between images. The dataset being used was 11K Hands dataset, which contains hand images of various subjects. The hand images may be of palmar side, dorsal side, left hand, right hand.

The project was completed into 3 different phases. Below is the list of tasks included in each of these phases.

### Phase 1

This phase included feature extraction using one of the following techniques

* Color Moments
* Local Binary Patterns
* Histogram of Oriented Gradients
* Scale Invariant Feature Transform

Visualized K similar images to the given input images based on these features.

### Phase 2

This phase included extracting latent semantics and visualizing similar images based on few dimensionality reduction techniques and labels from the metadata, using the four feature models discussed in Phase 1.

The few dimensionality reduction techniques are:

* Principal Component Analysis (PCA)
* Singular Value Decomposition (SVD)
* Non Negative Matrix Factorization (NMF)
* Latent Dirichlet Analysis (LDA)

The labels for the image that were utilized were: 

* Left hand vs Right hand
* Dorsal vs Palmar
* With Accessories vs Without Accessories
* Male vs Female

### Phase 3

This phase included implementing classifiers for images on "dorsal vs palmar" labels and relevance feedback system based on those classifiers.

The classifiers implemented were:

* Support Vector Machines
* Decision Tree Classifier
* Personalized Pagerank Classifier
