#! /usr/bin/env python3
import argparse
from project_utils import *
from dimensionality_reduction import *
from feature_extraction import *
import numpy as np
from tqdm import tqdm
import matplotlib
import cv2
import os
# from pprint import pprint

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--model", help="Name of the model to use - CM, LBP, HOG, SIFT", type=str, choices=["CM", "LBP", "HOG", "SIFT"], required=True)
    parser.add_argument("--lsa_model", help="Name of the latent semantic analysis model to use - PCA, SVD, NMF or LDA", type=str, choices=["PCA", "LDA", "NMF", "SVD"], required=True)
    parser.add_argument("--k_features", help="Number of k latent semantics to use", type=int, required=True)
    # parser.add_argument("--get_v", help="Execute extra credit?", type=bool, required=True)
    parser.add_argument("--top_latent", help="Execute extra credit?", type=int, required=True)
    parser.add_argument("--top_img", help="Execute extra credit?", type=int, required=True)

    return parser

def get_latent_feature_vs_imageIds_matrix(reduced_data_ids, k_features):
    feature_image_matrix = []
    feature_score_matrix = []
    for column_idx in tqdm(range(1,k_features + 1)):
        # sorted_mat= reduced_data_ids[reduced_data_ids[:,column_idx].argsort(kind='stable')].T
        sorted_mat = np.array(sorted(reduced_data_ids.tolist(), key=lambda a_entry: float(a_entry[column_idx]))).T

        feature_image_matrix.append(np.flip(sorted_mat[0]))
        feature_score_matrix.append(np.flip(sorted_mat[column_idx]))
    # pprint(feature_score_matrix)
    # print(np.array(feature_image_matrix).shape)
    return feature_image_matrix, feature_score_matrix

#feature_image_matrix is an array
#as under
# [
#  [image_ids sorted in order of latent semantic1]
#  [image_ids sorted in order of latent semantic2]
#  [image_ids sorted in order of latent semantic3]
#  .
#  .
#  .
#  .

 #]

def get_dot_prod_mat(reduced_data, V_matrix, image_ids):
    print(reduced_data.shape)
    print(np.array(V_matrix).shape)
    dot_prod_matrix_ids = np.concatenate((np.array(image_ids).reshape(200,1),np.matmul(reduced_data, np.array(V_matrix))),axis=1)
    return dot_prod_matrix_ids

def get_highest_dot_prod_images(dot_prod_matrix_ids, top_latent):
    images_with_highest_dot_prod=[]
    for column_idx in range(1, top_latent + 1):
        # sorted_mat = dot_prod_matrix_ids[dot_prod_matrix_ids[:,column_idx].argsort()].T
        sorted_mat = np.array(sorted(dot_prod_matrix_ids.tolist(), key=lambda a_entry: float(a_entry[column_idx]))).T

        images_with_highest_dot_prod.append(sorted_mat[0][-1])

    return images_with_highest_dot_prod



def create_subplots(feature_image_matrix, image_folder, feature_score_matrix):
    columns = len(feature_image_matrix[0])
    rows = len(feature_image_matrix)
    f, s_arr = plt.subplots(rows, columns, figsize=(8, 8))
    f.suptitle("Latent Features")
    for col_num, ax in enumerate(s_arr[0]):
        ax.set_title("{0}".format(col_num))

    for i in tqdm(range(rows)):
        for j in range(columns):
            s_arr[i][j].title.set_text("{0:.2f}".format(float(feature_score_matrix[i][j])))
            s_arr[i][j].axis("off")
            s_arr[i][j].imshow(cv2.cvtColor(cv2.imread(os.path.join(image_folder,feature_image_matrix[i][j])), cv2.COLOR_BGR2RGB))
    plt.show()

def create_matplot(image_list, image_folder):
    k = len(image_list)
    rows = int(np.sqrt(k)) + 1
    cols = int(np.ceil(k/rows))
    f, s_arr = plt.subplots(rows, cols, figsize=(8, 8))
    f.suptitle("Highest Dot product with ")
    for col_num, ax in enumerate(s_arr[0]):
        ax.set_title("{0}".format(col_num))
    count=0
    for i in tqdm(range(rows)):
        for j in range(cols):
            print(count)
            s_arr[i][j].title.set_text("latent semantic{0}".format(count))
            s_arr[i][j].axis("off")
            s_arr[i][j].imshow(cv2.cvtColor(cv2.imread(os.path.join(image_folder,image_list[count])), cv2.COLOR_BGR2RGB))
            count+=1

    # plt.show()

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()

    data_matrix, image_ids = get_data_matrix(args.model)
    reduced_data, image_ids = dimensionality_reduction(data_matrix, image_ids, args, viz=False)
    # reduced_data= reduced_data.astype(np.int8)
    reduced_data = np.around(reduced_data, decimals=3)
    reduced_data_ids = np.concatenate((np.array(image_ids).reshape(200,1), reduced_data),axis=1)
    # a[a[:,0].argsort()] use thisfor understanding how I sort a 2D matrix

    feature_image_matrix, feature_score_matrix = get_latent_feature_vs_imageIds_matrix(reduced_data_ids, args.k_features)
    #truncate the matrix for visualization
    feature_image_matrix = np.asarray(feature_image_matrix)[:args.top_img, :args.top_latent]
    feature_score_matrix = np.asarray(feature_score_matrix)[:args.top_img, :args.top_latent]
    create_subplots(feature_image_matrix, args.image_folder, feature_score_matrix)

    V_matrix = get_V_matrix(data_matrix, image_ids, args, viz=False) #k*len_of_image_feature_vec
    dot_prod_matrix_ids = get_dot_prod_mat(data_matrix, V_matrix, image_ids)
    images_with_highest_dot_prod = get_highest_dot_prod_images(dot_prod_matrix_ids, args.top_latent)
    create_matplot(images_with_highest_dot_prod, args.image_folder)
    plt.show()

if __name__ == "__main__":
    main()

# python3 extra_cred.py --image_folder /home/dhruv/Allprojects/MWDB/Hand_small --metadata_file /home/dhruv/Allprojects/MWDB/HandInfo.csv --model CM --lsa_model PCA --k_features 20 --top_latent 5 --top_img 5 --extra_cred True
