
#! /usr/bin/env python3
import argparse
from project_utils import *
from feature_extraction import *
from distance_measures import find_similar_images
# from tqdm import tqdm
import csv
import pandas as pd
from sklearn import svm
def experiment():
    features = ["CM", "LBP","HOG","SIFT"]
    reducers = ["SVD","PCA","NMF","LDA"]
    label_list = ["left-hand", "right-hand", "dorsal", "palmar", "with accessories", "without accessories", "male","female"]
    for label in label_list:
        for feature in features:
            for reducer in reducers:
                if feature == "CM" and reducer == "NMF":
                    continue
                returned_label, true_label = main(feature,reducer,label)
                if returned_label == true_label:
                    classification = "Correct"
                else:
                    classification = "Wrong"
                print("Result: " + returned_label + " / " + true_label)
                list1 = [feature, reducer, true_label, returned_label, classification]
                with open("task5ex.csv", "a") as fp:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(list1)

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--model", help="Name of the model to use - CM, LBP, HOG, SIFT", type=str, choices=["CM", "LBP", "HOG", "SIFT"], required=True)
    parser.add_argument("--lsa_model", help="Name of the latent semantic analysis model to use - PCA, SVD, NMF or LDA", type=str, choices=["PCA", "LDA", "NMF", "SVD"], required=True)
    parser.add_argument("--k_features", help="Number of k latent semantics to use", type=int, required=True)
    parser.add_argument("--label", help="Labels for the image", type=str, required=True, choices=["left-hand", "right-hand", "dorsal", "palmar", "with accessories", "without accessories", "male", "female"])
    parser.add_argument("--query_image_path", help="Path to the query image to classify", type=str, required=True)
    return parser

def get_distance_measures_for_label(args, filter_string, feature_vector, query_image_id):
    data_matrix, image_ids = get_data_matrix(args.model, filter_string)
    if args.model == "SIFT":
        # Sift needs special hand-holding here, as it is slightly tricky.
        max_vector_length = data_matrix.shape[1]
        sift_vectors = []
        for feature in feature_vector:
            if len(sift_vectors) == max_vector_length:
                break
            sift_vectors.extend(feature)
        new_data_matrix = np.append(data_matrix, np.array(sift_vectors).reshape(1, len(sift_vectors)), axis = 0)
    else:
        new_data_matrix = np.append(data_matrix, np.array(feature_vector).reshape(1, len(feature_vector)), axis = 0)
    image_ids.append(query_image_id)
    dimensionality_reduction(new_data_matrix, image_ids, args, args.label)

    similar_images = find_similar_images(query_image_id, args.model, len(image_ids) - 1, args.label)
    distance_score = 0
    for similar_image in similar_images:
        distance_score += similar_image["distance_score"]
    return distance_score/(len(image_ids) - 1)

def find_threshold(args, filter_string, feature_vector, query_image_id):
    data_matrix, image_ids = get_data_matrix(args.model, filter_string)
    if args.model == "SIFT":
        # Sift needs special hand-holding here, as it is slightly tricky.
        max_vector_length = data_matrix.shape[1]
        sift_vectors = []
        for feature in feature_vector:
            if len(sift_vectors) == max_vector_length:
                break
            sift_vectors.extend(feature)
        new_data_matrix = np.append(data_matrix, np.array(sift_vectors).reshape(1, len(sift_vectors)), axis = 0)
    else:
        new_data_matrix = np.append(data_matrix, np.array(feature_vector).reshape(1, len(feature_vector)), axis = 0)
    image_ids.append(query_image_id)
    dimensionality_reduction(new_data_matrix, image_ids, args, args.label)
    max_threshold = float("-inf")
    for image_id in image_ids:
        similar_images = find_similar_images(image_id, args.model, len(image_ids) - 1, args.label)
        distance_score = 0
        for similar_image in similar_images:
            distance_score += similar_image["distance_score"]
        max_threshold = max(max_threshold, distance_score/(len(image_ids) - 1))
    return max_threshold

def get_converse_label(label):
    if label == "left-hand":
        return "right-hand"
    elif label == "right-hand":
        return "left-hand"
    elif label == "dorsal":
        return "palmar"
    elif label == "palmar":
        return "dorsal"
    elif label == "with accessories":
        return "without accessories"
    elif label == "without accessories":
        return "with accessories"
    elif label == "male":
        return "female"
    elif label == "female":
        return "male"

def get_lable_type(label):
    if label == "left-hand":
        return "left-right"
    elif label == "right-hand":
        return "left-right"
    elif label == "dorsal":
        return "side"
    elif label == "palmar":
        return "side"
    elif label == "with accessories":
        return "accessories"
    elif label == "without accessories":
        return "accessories"
    elif label == "male":
        return "gender"
    elif label == "female":
        return "gender"

def get_true_label(query_image_path, metadata_file, label_type):
    query_image_id = query_image_path.split("/")[-1]
    full_dataset = pd.read_csv(metadata_file)
    new_df=full_dataset.loc[full_dataset['imageName']== query_image_id,['gender','accessories', 'aspectOfHand']]
    if label_type == "gender":
        return new_df['gender'].values[0]
    elif label_type == "accessories":
        if new_df['accessories'].values[0] == 0:
            return "without accessories"
        else:
            return "with accessories"
    elif label_type == "side":
        return new_df['aspectOfHand'].values[0].split(' ')[0]
    elif label_type == "left-right":
        if new_df['aspectOfHand'].values[0].split(' ')[1] == 'left':
            return 'left-hand'
        else:
            return 'right-hand'

def main(model, lsa_model, label):
    parser = setup_arg_parse()
    args = parser.parse_args()

    args.model = model
    args.lsa_model = lsa_model
    args.label = label
    #original_metadata_file = '/Users/sparkexel/Desktop/Class/Fall2019/mwdb/Assignment2/MWDBAssignment2/testdata/queryhands_metadata.csv'
    original_metadata_file = '~/ASU/MWDB/Project/DataSet/HandInfo.csv'

    if not os.path.exists(args.query_image_path):
        print("Invalid query image, please check the path")
        os.exit(-1)

    query_image_id = os.path.basename(args.query_image_path)
    if args.model == "CM":
        feature_vector = extract_color_moments(args.query_image_path)
    elif args.model == "LBP":
        feature_vector = extract_lbp_features(args.query_image_path)
    elif args.model == "HOG":
        feature_vector = extract_hog_features(args.query_image_path)
    elif args.model == "SIFT":
        feature_vector = extract_sift_features(args.query_image_path)

    filter_string = convert_label_to_filterstring(args.label)
    data_matrix, image_ids = get_data_matrix(args.model, filter_string)
    if args.model == "SIFT":
        # Sift needs special hand-holding here, as it is slightly tricky.
        max_vector_length = data_matrix.shape[1]
        sift_vectors = []
        for feature in feature_vector:
            if len(sift_vectors) == max_vector_length:
                break
            sift_vectors.extend(feature)
        new_data_matrix = np.append(data_matrix, np.array(sift_vectors).reshape(1, len(sift_vectors)), axis = 0)
    else:
        new_data_matrix = np.append(data_matrix, np.array(feature_vector).reshape(1, len(feature_vector)), axis = 0)
    image_ids.append(query_image_id)
    reduced_data_matrix, image_ids = dimensionality_reduction(new_data_matrix, image_ids, args, args.label)
    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)
    oc_svm_clf.fit(reduced_data_matrix)
    pred=oc_svm_clf.predict(reduced_data_matrix[-1].reshape(1,-1))
    # distance_score = get_distance_measures_for_label(args, filter_string, feature_vector, query_image_id)
    # similarity_score = (1 / distance_score) * 100
    # Threshold_score = find_threshold(args, filter_string, feature_vector, query_image_id)
    # print("distance score for {0} is {1}".format(args.label, distance_score))
    # print("Similarity score for {0} is {1}".format(args.label, similarity_score))
    # print("Threshold_score score for {0} is {1}".format(args.label, Threshold_score))

    if pred:
        print("The given image {0} belongs to class {1}".format(query_image_id, args.label))
        returned_label = args.label
    else:
        print("The given image {0} belongs to class {1}".format(query_image_id, get_converse_label(args.label)))
        returned_label = get_converse_label(args.label)

    # Delete the Query image from the database
    conn = connect_to_db()
    conn.mwdb_project.image_features.delete_one({'imageName': query_image_id})
    conn.close()
    true_label = get_true_label(query_image_id, original_metadata_file, get_lable_type(label))
    return returned_label, true_label

if __name__ == "__main__" :
    # main()
    experiment()
    #main()
