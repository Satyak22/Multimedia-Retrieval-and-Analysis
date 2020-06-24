from project_utils import *
import numpy as np
from multiprocessing import Pool

# finds similarity between input image and all images in database and inserts into the database.
# uses Euclidean distance. It varies slightly for sift and hog
def calculate_distance_sift(image_id_dict):
    query_image_id = image_id_dict["query_image_id"]
    image_id = image_id_dict["imageName"]
    mongo_Client = connect_to_db()
    DB_HANDLE = mongo_Client.mwdb_project
    query_feature_vector = np.asarray(DB_HANDLE.image_features.find_one({'imageName' : query_image_id})['SIFT_reduced'])
    image_feature_vector = np.asarray(DB_HANDLE.image_features.find_one({'imageName' : image_id})['SIFT_reduced'])

    n = query_feature_vector.shape[0]
    img_distance = 0
    for vector in query_feature_vector:
        img_distance += np.min(np.sqrt(np.sum(np.square(image_feature_vector - vector), axis = 0)))
    img_distance /= n

    DB_HANDLE.image_features.update_one({'imageName': image_id}, {'$set' : {"distance_score" : img_distance}})
    mongo_Client.close()

def calculate_distance_hog(image_id_dict):
    query_image_id = image_id_dict["query_image_id"]
    image_id = image_id_dict["imageName"]
    mongo_Client = connect_to_db()
    DB_HANDLE = mongo_Client.mwdb_project
    query_feature_vector = np.asarray(DB_HANDLE.image_features.find_one({'imageName' : query_image_id})['HOG_reduced'])
    image_feature_vector = np.asarray(DB_HANDLE.image_features.find_one({'imageName' : image_id})['HOG_reduced'])

    img_distance = np.sqrt(np.sum(np.square(image_feature_vector - query_feature_vector)))

    DB_HANDLE.image_features.update_one({'imageName': image_id}, {'$set' : {"distance_score" : img_distance}})
    mongo_Client.close()

# distance/similarity function for color moments. Uses eucledian distance.
def calculate_distance_cm(image_id_dict):
    idx_image_id = image_id_dict["query_image_id"]
    image_id = image_id_dict["imageName"]
    mongo_Client = connect_to_db()
    DB_HANDLE = mongo_Client.mwdb_project
    query_image_color_moments = DB_HANDLE.image_features.find_one({'imageName' : idx_image_id})['CM_reduced']
    image_color_moments = DB_HANDLE.image_features.find_one({'imageName' : image_id})['CM_reduced']

    distance_score = 0
    for q, i in zip(query_image_color_moments, image_color_moments):
        distance_score += np.square(q - i)
    distance_score = np.sqrt(distance_score)

    DB_HANDLE.image_features.update_one({'imageName': image_id}, {'$set' : {"distance_score" : distance_score}})
    mongo_Client.close()

# Distance/similarity function for LBP, uses eucledian distance between histograms.
def calculate_distance_lbp(image_id_dict):
    idx_image_id = image_id_dict["query_image_id"]
    image_id = image_id_dict["imageName"]
    mongo_Client = connect_to_db()
    DB_HANDLE = mongo_Client.mwdb_project
    query_lbp = DB_HANDLE.image_features.find_one({'imageName' : idx_image_id})['LBP_reduced']
    image_lbp = DB_HANDLE.image_features.find_one({'imageName' : image_id})['LBP_reduced']
    distance_score = 0
    for query_lbp_feature, image_lbp_feature in zip(query_lbp, image_lbp):
        distance_score += np.sqrt(np.square(query_lbp_feature - image_lbp_feature))

    DB_HANDLE.image_features.update_one({'imageName': image_id}, {'$set' : {"distance_score" : distance_score}})
    mongo_Client.close()

# Driver method that parallely computes the distance score between the given query image and other images in the db.
def find_similar_images(image_id, model, k, label = None):
    mongo_Client = connect_to_db()
    DB_HANDLE = mongo_Client.mwdb_project
    image_collection = DB_HANDLE.image_features
    image_collection.create_index([('distance_score', 1)])
    image_collection.update_many({}, {'$set' : {"query_image_id": image_id, "distance_score" : 100000000000 }})
    if label == None:
        image_ids = list(image_collection.find({}, {"imageName" : 1 , "_id": 0, "query_image_id" : 1 }))
    else:
        image_ids = list(image_collection.find({'label' : label}, {"imageName" : 1 , "_id": 0, "query_image_id" : 1 }))
    pool = Pool()
    if model == "CM":
        pool.map(calculate_distance_cm, image_ids)
    elif model == "LBP":
        pool.map(calculate_distance_lbp, image_ids)
    elif model == "HOG":
        pool.map(calculate_distance_hog, image_ids)
    elif model == "SIFT":
        pool.map(calculate_distance_sift, image_ids)
    pool.close()
    pool.join()
    images = list(DB_HANDLE.image_features.find({"imageName" :{"$ne" : image_id}}, {"imageName": 1, "_id" : 0, "distance_score" : 1, "image_path": 1}).sort("distance_score", 1).limit(k))
    mongo_Client.close()
    return images
