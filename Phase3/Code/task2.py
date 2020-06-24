from collections import Counter
import pprint
from project_utils import *
import argparse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from task1 import populate_database, process_all_images
style.use('ggplot')

#! /usr/bin/env python3


class K_Means:
    def __init__(self, k=5, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [
                    np.linalg.norm(
                        featureset -
                        self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) /
                          original_centroid * 100.0) > self.tol:
                    # print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification, min(distances)


def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder",
        help="Path to the folder containing images",
        type=str,
        required=True)
    parser.add_argument("--test_folder", help="Path to the folder containing the test images", type=str, required=True)
    parser.add_argument(
        "--train_metadata",
        help="Path to the metadata file",
        type=str,
        required=True)
    parser.add_argument(
        "--c",
        help="enter number of clusters",
        type=int,
        required=True)
    parser.add_argument("--labels_csv", help="Path to the CSV file that contains the labels for unlabelled images, used for accuracy calculation", type=str)
    return parser


def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    populate_database(args)
    process_all_images(args.test_folder)
    args.label = "palmar"
    args.lsa_model = "SVD"
    args.model = "CM"
    args.k_features = 20
    filterstring = convert_label_to_filterstring(args.label)
    mongo_client = connect_to_db()
    images = list(
        mongo_client.mwdb_project.image_features.find(
            filterstring, {
                "imageName": 1}))
    mongo_client.close()
    print(
        "There are {0} images in the db matching the label - {1}".format(len(images), args.label))
    data_matrix_palmar, image_ids_palmar = get_data_matrix(
        args.model, filterstring)
    # reduced_dims_palmar, image_ids_palmar = dimensionality_reduction(data_matrix_palmar, image_ids_palmar, args, args.label, viz=False)

    args.label = "dorsal"
    filterstring = convert_label_to_filterstring(args.label)
    mongo_client = connect_to_db()
    images = list(
        mongo_client.mwdb_project.image_features.find(
            filterstring, {
                "imageName": 1}))
    mongo_client.close()
    print(
        "There are {0} images in the db matching the label - {1}".format(len(images), args.label))
    data_matrix_dorsal, image_ids_dorsal = get_data_matrix(
        args.model, filterstring)
    # reduced_dims_dorsal, image_ids_dorsal = dimensionality_reduction(data_matrix_dorsal, image_ids_dorsal, args, args.label, viz=False)

    full_reduced_dims, V_matrix = get_V_matrix(
        np.concatenate(
            (data_matrix_dorsal, data_matrix_palmar), axis=0), np.concatenate(
            (image_ids_dorsal, image_ids_palmar), axis=0), args, label=None, viz=False)
    # full_reduced_dims, full_ids = dimensionality_reduction(np.concatenate((data_matrix_dorsal, data_matrix_palmar),axis=0), np.concatenate((image_ids_dorsal,image_ids_palmar),axis=0), args, args.label, viz=False)
    reduced_dims_dorsal = full_reduced_dims[:data_matrix_dorsal.shape[0]]
    reduced_dims_palmar = full_reduced_dims[data_matrix_dorsal.shape[0]:]

    target_paths, target_image_ids = enumerate_files_in_dir(args.test_folder)
    target_matrix, target_image_ids = get_target_matrix(target_image_ids, args.model, mongo_client)

    target_reduced_dims = np.matmul(target_matrix, V_matrix)
    # print(target_reduced_dims.shape)

    # print(reduced_dims_dorsal)
    # print(reduced_dims_palmar)

    def create_2_spatial_dim_planes(reduced_dims, queries, space, c):
        colors = 10 * ["g", "r", "c", "b", "k"]

        clf = K_Means(k=c)
        clf.fit(reduced_dims)
        fig = plt.figure(figsize=(32, 32))
        fig.patch.set_facecolor('xkcd:black')
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')

        def create_plots(ax, dim_num):
            ax.set_title('{} latent space plot for dimensions {},{},{}'.format(
                space, dim_num + 0, dim_num + 1, dim_num + 2), color="gray")
            ax.set_facecolor('black')
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.zaxis.set_major_locator(plt.MaxNLocator(3))
            ax.grid(False)
            ax.set_xlim3d([reduced_dims.min(axis=0)[dim_num + 0],
                           reduced_dims.max(axis=0)[dim_num + 0]])
            ax.set_ylim3d([reduced_dims.min(axis=0)[dim_num + 1],
                           reduced_dims.max(axis=0)[dim_num + 1]])
            ax.set_zlim3d([reduced_dims.min(axis=0)[dim_num + 2],
                           reduced_dims.max(axis=0)[dim_num + 2]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # plt.show()

            # plt.ylim(reduced_dims.min(axis=0)[1], reduced_dims.max(axis=0)[1])
            # plt.xlim(reduced_dims.min(axis=0)[0],reduced_dims.max(axis=0)[0])

            for centroid in clf.centroids:
                ax.scatter(clf.centroids[centroid][dim_num +
                                                   0], clf.centroids[centroid][dim_num +
                                                                               1], clf.centroids[centroid][dim_num +
                                                                                                           2], c='white', marker='o', s=150, linewidths=1)
            # plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
            #             marker="o", color="k", s=150, linewidths=1)

            for classification in clf.classifications:
                color = colors[classification]
                for featureset in clf.classifications[classification]:
                    ax.scatter(featureset[dim_num + 0],
                               featureset[dim_num + 1],
                               featureset[dim_num + 2],
                               c=color,
                               marker='x',
                               s=15,
                               linewidths=1)
                    # plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=15, linewidths=1)
        try:
            create_plots(ax, 0)
            create_plots(ax2, 1)
            create_plots(ax3, 2)
            create_plots(ax4, 3)
        except BaseException:
            pass
        # return clf.predict(query)#need to compress this query using V_matrix
        # print(clf.predict(query))
        ans = []
        for query in queries:
            ans.append(clf.predict(query))
        return ans

    ans_dorsal = create_2_spatial_dim_planes(
        reduced_dims_dorsal,
        target_reduced_dims,
        "dorsal",
        args.c)  # query = reduced_dims_dorsal[-1]  "dorsal" str is just used for plot title
    ans_palmar = create_2_spatial_dim_planes(
        reduced_dims_palmar,
        target_reduced_dims,
        "palmar",
        args.c)  # query = reduced_dims_dorsal[-1]	"dorsal" str is just used for plot title
    # if query_distance_dorsal<query_distance_palmar:
    # 	print("given query image is dorsal")
    # else:
    # 	print("given query image is palmar")

    # ans_dorsal and ans_palmar contains the elements [(cluster, distance),
    # (cluster, distance),...]
    result = []
    for idx in range(len(ans_dorsal)):
        if ans_dorsal[idx][-1] > ans_palmar[idx][-1]:
            result.append((target_image_ids[idx], "palmar"))
        else:
            result.append((target_image_ids[idx], "dorsal"))
# 	print(ans_dorsal)
# 	print(ans_palmar)
    print(result)
    if "labels_csv" in args:
        actuals = get_actual_labels_from_csv(args.labels_csv, target_image_ids)
        accuracy = get_accuracy(actuals, result)
        print("accuracy is {0}".format(accuracy))
    # plt.imsave()
    plt.show()


if __name__ == "__main__":
    main()
