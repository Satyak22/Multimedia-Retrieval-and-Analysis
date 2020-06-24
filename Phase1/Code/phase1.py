#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
from scipy.stats import skew
import time
from skimage.feature import local_binary_pattern
from multiprocessing import Pool
from pymongo import MongoClient
from matplotlib import pyplot as plt

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="Path to the folder containing images", type=str)
    parser.add_argument("task", help = "Name of the task to execute - 1, 2, or 3", type=int, choices=[1, 2, 3])
    parser.add_argument("--model", help="Name of the model to use - CM or LBP", type=str, choices=["CM", "LBP"])
    parser.add_argument("--image_id", help="ID of the image to compare against", type=str)
    parser.add_argument("--k", help="number of images to return", type=int)
    return parser

def calculate_first_moment(channel):
    return np.mean(channel)

def calculate_second_moment(channel):
    variance = np.var(channel)
    return np.sqrt(variance)

def calculate_third_moment(channel):
    # Flatten the array, and calculate the skew
    return skew(np.ndarray.flatten(channel), 0)

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

# Method that extracts the color moments, reads image from disk, converts to YUV,
# splits it into 100 x 100 windows then extracts color moments for each channel for each window.
def extract_color_moments(path_to_image):
    DB_HANDLE = connect_to_db()
    image_id, _ = os.path.splitext(os.path.basename(path_to_image))
    img = cv2.imread(path_to_image)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(img_yuv)
    color_moments = []
    for channel in [y, u, v]:
        channel_moments = []
        slices = slice_channel(channel)
        for channel_slice in slices:
            channel_moments.append(calculate_first_moment(channel_slice))
            channel_moments.append(calculate_second_moment(channel_slice))
            channel_moments.append(calculate_third_moment(channel_slice))
        color_moments.append(channel_moments)

    image_features = DB_HANDLE.image_features
    image = image_features.update_one({'image_id': image_id}, { '$set' : {'image_path' : path_to_image, 'ColorMoments' : color_moments}  }, upsert=True)
    return color_moments

# Method to extract Lbp histograms, read image from disk, conver to grayscale,
# split to 100 x 100 windows, compute LBP and then the histogram of the LBP image.
def extract_lbp_features(path_to_image):
    image_id, _ = os.path.splitext(os.path.basename(path_to_image))
    img = cv2.imread(path_to_image)
    img_greyscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    slices = slice_channel(img_greyscale)
    lbp_features = []
    for channel_slice in slices:
        # LBP Is calculated with radius = 1 and points  =  3, method = "default"
        lbp = local_binary_pattern(img_greyscale, 1, 3, 'default')
        hist, edges = np.histogram(lbp, bins = 256, range = (0.0,255.0))
        hist = hist.astype("float")
        hist = hist/(hist.sum() + 1e-7)
        lbp_features.append(hist.tolist())
    DB_HANDLE = connect_to_db()
    image = DB_HANDLE.image_features.update_one({'image_id': image_id}, {'$set' : { 'image_path' : path_to_image, 'LBP' : lbp_features }}, upsert=True)
    return lbp_features

# distance/similarity function for color moments. Uses eucledian distance.
def calculate_distance_cm(image_id_dict):
    idx_image_id = image_id_dict["query_image_id"]
    image_id = image_id_dict["image_id"]
    DB_HANDLE = connect_to_db()
    query_image_color_moments = DB_HANDLE.image_features.find_one({'image_id' : idx_image_id})['ColorMoments']
    image_color_moments = DB_HANDLE.image_features.find_one({'image_id' : image_id})['ColorMoments']
    distance_score = 0
    for idx_ch_moments, img_ch_moments in zip(query_image_color_moments, image_color_moments):
        for i in range(0, len(idx_ch_moments), 3):
            distance_score = distance_score + np.sqrt((idx_ch_moments[i] - img_ch_moments[i]) ** 2 + (idx_ch_moments[i+1] - img_ch_moments[i+1]) ** 2 + (idx_ch_moments[i+2] - img_ch_moments[i+2]) ** 2)

    DB_HANDLE.image_features.update_one({'image_id': image_id}, {'$set' : {"distance_score" : distance_score}})

# Distance/similarity function for LBP, uses eucledian distance between histograms.
def calculate_distance_lbp(image_id_dict):
    idx_image_id = image_id_dict["query_image_id"]
    image_id = image_id_dict["image_id"]
    DB_HANDLE = connect_to_db()
    query_lbp = DB_HANDLE.image_features.find_one({'image_id' : idx_image_id})['LBP']
    image_lbp = DB_HANDLE.image_features.find_one({'image_id' : image_id})['LBP']
    distance_score = 0
    for query_lbp_slice, image_lbp_slice in zip(query_lbp, image_lbp):
        slice_score = 0
        for query_val, image_val in zip(query_lbp_slice, image_lbp_slice):
            slice_score = slice_score + (query_val - image_val) ** 2
        distance_score = distance_score + np.sqrt(slice_score)

    DB_HANDLE.image_features.update_one({'image_id': image_id}, {'$set' : {"distance_score" : distance_score}})

# Driver method that parallely computes the distance score between the given query image and other images in the db.
def find_similar_images(image_id, model, k):
    DB_HANDLE = connect_to_db()
    image_collection = DB_HANDLE.image_features
    image_collection.update_many({}, {'$set' : {"query_image_id": image_id, "distance_score" : -1}})
    image_ids = list(image_collection.find({}, {"image_id" : 1 , "_id": 0, "query_image_id" : 1 }))
    pool = Pool()
    if model == "CM":
        pool.map(calculate_distance_cm, image_ids)
    elif model == "LBP":
        pool.map(calculate_distance_lbp, image_ids)
    pool.close()
    pool.join()
    images = list(DB_HANDLE.image_features.find({"image_id" :{"$ne" : image_id}}, {"image_id": 1, "_id" : 0, "distance_score" : 1, "image_path": 1}).sort("distance_score", 1).limit(k))
    return images

# Helper methods to enumerate all files in the directory
def enumerate_files_in_dir(image_path):
    directory = os.fsencode(image_path)
    full_paths = []
    for image in os.listdir(directory):
        # TODO: Add check for image
        full_path = os.fsdecode(os.path.join(directory, image))
        full_paths.append(full_path)
    return full_paths

# Driver Method that parallelly extracts the features of all images in a directory, based on the specified model
def process_all_images(image_path, model):
    full_paths = enumerate_files_in_dir(image_path)
    print("There are {0} images in the directory".format(len(full_paths)))

    pool = Pool()
    if model == "CM":
        pool.map(extract_color_moments, full_paths)
    elif model == "LBP":
        pool.map(extract_lbp_features, full_paths)
    pool.close()
    pool.join()
    return

def connect_to_db():
    client = MongoClient()
    db = client.mwdb_project
    return db

# Driver for task 1
def do_task_1(args):
    if "image_id" not in args:
        print("Image ID is mandatory for task 1")
        os.exit(-1)
    filepath = os.path.join(args.image_folder, "{0}.jpg".format(args.image_id))
    print("Processing image at {0}".format(filepath))
    color_moments = extract_color_moments(filepath)
    print("The color moments are:")
    print(color_moments)

    lbp_features = extract_lbp_features(filepath)
    print("The lbp features are:")
    print(lbp_features)
    return

# Driver for task 2
def do_task_2(args):
    if args.model is not None:
        print("Extracting {0} of images in {1}".format(args.model, args.image_folder))
        process_all_images(args.image_folder, args.model)
    else:
        print("Extracting CM of images in {0}".format(args.image_folder))
        process_all_images(args.image_folder, "CM")
        print("Extracting LBP of images in {0}".format(args.image_folder))
        process_all_images(args.image_folder, "LBP")
    print("Extraction complete, please inspect database.")
    return

# Driver for task 3
def do_task_3(args):
    if "k" not in args:
        print("K not specified")
        os.exit(-1)

    if "model" not in args:
        print("Model not specified")
        os.exit(-1)

    rows = connect_to_db().image_features.count_documents({'ColorMoments': { '$exists': True }, 'LBP' :  { '$exists' : True }})
    all_images = enumerate_files_in_dir(args.image_folder)
    if rows != len(all_images):
        do_task_2(args)
    similar_images = find_similar_images(args.image_id, args.model, args.k)
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("K similar images")
    rows = int(np.sqrt(len(similar_images)))
    columns = len(similar_images) // rows
    for i in range(0, args.k):
        fig.add_subplot(rows + 1, columns, i + 1).title.set_text( str(similar_images[i]["image_id"]) + " : " + str(similar_images[i]["distance_score"]))
        plt.imshow(cv2.cvtColor(cv2.imread(similar_images[i]["image_path"]), cv2.COLOR_BGR2RGB))
    fig.add_subplot(rows + 1, columns, i + 2).title.set_text("Query Image")
    plt.imshow(cv2.cvtColor(cv2.imread(os.path.join(args.image_folder, args.image_id + ".jpg")), cv2.COLOR_BGR2RGB))
    plt.show()
    for similar_image in similar_images:
        print(similar_image)

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    if args.task == 1:
        do_task_1(args)
    elif args.task == 2:
        do_task_2(args)
    else:
        do_task_3(args)


if __name__ == "__main__" :
    main()
