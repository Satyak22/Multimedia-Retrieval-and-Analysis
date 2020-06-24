import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from matplotlib import pyplot as plt
from pymongo import MongoClient
from dimensionality_reduction import reduce_dimensions_pca, reduce_dimensions_svd, reduce_dimensions_lda, reduce_dimensions_nmf
from collections import Counter

def plot_results(similar_images, query_image_path):
    k = len(similar_images)
    # this part is for displaying 'k' similar images using matplotlib library
    rows = int(np.sqrt(k)) + 1
    cols = int(np.ceil(k/rows))
    if rows * cols <= k:
        cols += 1
    f, s_arr = plt.subplots(rows, cols)
    s_arr[0][0].axis("off")
    s_arr[0][0].text(0.5,-0.1, "Target Image", size=6, ha="center", transform=s_arr[0][0].transAxes)
    s_arr[0][0].imshow(plt.imread(query_image_path))
    i,j = 0,1
    for x in similar_images:
        if k <= 0:
            break
        s_arr[i][j].axis("off")
        s_arr[i][j].text(0.5,-0.1,"Score: " + str(x["distance_score"]) + "; " + x["imageName"], size=6, ha="center", transform=s_arr[i][j].transAxes)
        s_arr[i][j].imshow(plt.imread(x["image_path"]))
        j += 1
        if j >= cols:
            j = 0
            i += 1
        k -= 1
    while i < rows:
        while j < cols:
            s_arr[i][j].axis("off")
            j += 1
        i += 1
        j = 0

    plt.show()

# Slices the given image into 100 x 100 windows
def slice_channel(channel):
    slices = []
    r = 0
    while r < channel.shape[0]:
        c = 0
        while c < channel.shape[1]:
            channel_slice = channel[r:r + 100, c:c + 100]
            slices.append(channel_slice)
            c = c + 100
        r = r + 100
    return slices

# Helper methods to enumerate all files in the directory
def enumerate_files_in_dir(image_path):
    directory = os.fsencode(image_path)
    full_paths = []
    image_ids = []
    for image in os.listdir(directory):
        if os.fsdecode(image).endswith(".jpg"):
            image_ids.append(os.fsdecode(image))
            full_path = os.fsdecode(os.path.join(directory, image))
            full_paths.append(full_path)
    return full_paths, image_ids

def connect_to_db():
    client = MongoClient()
    return client

def convert_data_matrix_cmlda(data_matrix):
    bias = abs(int(np.min(data_matrix)))
    data=[]
    data_counter_matrix=[]
    for doc in np.array(data_matrix).astype(int):
        data_counter_matrix.append(list(Counter(doc).items()))

    new_corpus=[[0 for _ in range(255+bias)] for _ in range(len(data_counter_matrix))]
    for idx,element in enumerate(data_counter_matrix):
        pairs=[]
        for i,j in list(element):
            new_corpus[idx][int(i)]=int(j)

    return new_corpus#.T[~np.all(new_corpus.T == 0, axis=1)].T #removes blank columns


def transform_sift(data):
    minlen = min(map(len, data))
    data_transformed = []
    for row in data:
        row_data = []
        for i in range(minlen):
            row_data.extend(row[i])
        data_transformed.append(np.array(row_data))
    data_matrix = np.array(data_transformed)
    return data_matrix


def get_data_matrix(model, filterstring=None, flag = None):
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    if filterstring is None:
        images = db_handle.image_features.find({}, {"imageName" : 1, model : 1, "_id" : 0})
    else:
        images = db_handle.image_features.find(filterstring, {"imageName" :1, model :1, "_id" : 0})
    data = []
    image_ids = []
    for image in images:
        data.append(np.array(image[model]))
        image_ids.append(image["imageName"])
    if model == "SIFT":
        if flag:
            return data, image_ids
        else:
            data_matrix = transform_sift(data)
            return data_matrix, image_ids

    data_matrix  = np.array(data)
    mongo_client.close()
    return data_matrix, image_ids

def insert_reduced_features(model, reduced_data, image_ids, label):
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    model_name = "{0}_reduced".format(model)
    if label == None:
        for image_id, reduced_feature in zip(image_ids, reduced_data.tolist()):
            image = db_handle.image_features.update_one({'imageName': image_id}, { '$set' : { model_name : reduced_feature } })
    else:
        for image_id, reduced_feature in zip(image_ids, reduced_data.tolist()):
            image = db_handle.image_features.update_one({'imageName': image_id}, { '$set' : { model_name : reduced_feature , 'label' : label} })
    mongo_client.close()

def dimensionality_reduction(data_matrix, image_ids, args, label=None, viz=False):
    if args.lsa_model == "SVD":
        reduced_data = reduce_dimensions_svd(data_matrix, args.k_features, image_ids, viz)
    elif args.lsa_model == "PCA":
        reduced_data = reduce_dimensions_pca(data_matrix, args.k_features, image_ids, viz)
    elif args.lsa_model == "NMF":
        if args.model == "CM":
            print("NMF is not applicable to Color Moments")
            return
        reduced_data = reduce_dimensions_nmf(data_matrix, args.k_features, image_ids, viz)
    elif args.lsa_model == "LDA":
        if args.model == "CM":
            print("LDA is applicable only to a modified Color Moments.")
            data_matrix_cmlda = convert_data_matrix_cmlda(data_matrix)
            reduced_data = reduce_dimensions_lda(data_matrix_cmlda, args.k_features, image_ids, viz)
        else:
            reduced_data = reduce_dimensions_lda(data_matrix, args.k_features, image_ids, viz)

    insert_reduced_features(args.model, reduced_data, image_ids, label)
    return reduced_data, image_ids


def get_V_matrix(data_matrix, image_ids, args, label=None, viz=False):
    if args.lsa_model == "SVD":
        V_matrix = reduce_dimensions_svd(data_matrix, args.k_features, image_ids, viz, get_v=True)
    elif args.lsa_model == "PCA":
        V_matrix = reduce_dimensions_pca(data_matrix, args.k_features, image_ids, viz, get_v=False)
    elif args.lsa_model == "NMF":
        if args.model == "CM":
            print("NMF is not applicable to Color Moments")
            return
        V_matrix = reduce_dimensions_nmf(data_matrix, args.k_features, image_ids, viz, get_v=True)
    elif args.lsa_model == "LDA":
        if args.model == "CM":
            print("LDA is NOT applicable because of the Bag Of Word model.")
            # data_matrix_cmlda = convert_data_matrix_cmlda(data_matrix)
            # V_matrix = reduce_dimensions_lda(data_matrix_cmlda, args.k_features, image_ids, viz, get_v=True)
            return
        else:
            V_matrix = reduce_dimensions_lda(data_matrix, args.k_features, image_ids, viz, get_v=True)

    # insert_reduced_features(args.model, reduced_data, image_ids, label)
    return V_matrix


def ex_dimensionality_reduction(data_matrix, image_ids, model, lsa_model, k_features, label=None, viz=False):
    if lsa_model == "SVD":
        reduced_data = reduce_dimensions_svd(data_matrix, k_features, viz)
    elif lsa_model == "PCA":
        reduced_data = reduce_dimensions_pca(data_matrix, k_features, viz)
    elif lsa_model == "NMF":
        if model == "CM":
            print("NMF is not applicable to Color Moments")
            return
        reduced_data = reduce_dimensions_nmf(data_matrix, k_features, viz)
    elif lsa_model == "LDA":
        if model == "CM":
            print("LDA is modified and applicable to Color Moments")
            data_matrix_cmlda = convert_data_matrix_cmlda(data_matrix)
            reduced_data = reduce_dimensions_lda(data_matrix_cmlda, k_features, viz)
        else:
            reduced_data = reduce_dimensions_lda(data_matrix, k_features, viz)

    insert_reduced_features(model, reduced_data, image_ids, label)
    if extra_credit:
         return reduced_data, image_ids

def convert_label_to_filterstring(label):
    filterstring = {}
    if label == "left-hand":
        filterstring["aspectOfHand"] = { "$regex" : "left"}
    elif label == "right-hand":
        filterstring["aspectOfHand"] = { "$regex" : "right"}
    elif label == "dorsal":
        filterstring["aspectOfHand"] = { "$regex" : "dorsal"}
    elif label == "palmar":
        filterstring["aspectOfHand"] = { "$regex" : "palmar"}
    elif label == "with accessories":
        filterstring["accessories"] = 1
    elif label == "without accessories":
        filterstring["accessories"] = 0
    elif label == "male":
        filterstring["gender"] = "male"
    elif label == "female":
        filterstring["gender"] = "female"

    return filterstring

