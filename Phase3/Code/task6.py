#! /usr/bin/env python3
import argparse
from LSH import ImageLSH
from project_utils import connect_to_db, plot_results
from feature_extraction import extract_hog_features
from pprint import pprint
import numpy as np
from classifiers import DecisionTreeClassifier, SupportVectorMachine, rbf_kernel
from project_utils import get_data_matrix, convert_label_to_filterstring, enumerate_files_in_dir
from feature_extraction import extract_hog_features
from dimensionality_reduction import reduce_dimensions_lda
from task3 import *
import json

def get_modified_jump_matrix(feedbacks, image_ids, prev_jump_matrix):
    for feedback in feedbacks:
        if feedback[1] == "I":
            jump_val = -1
        elif feedback[1] == "R":
            jump_val = 1
        prev_jump_matrix[image_ids.index(feedback[0])] += jump_val
    new_jump_matrix = prev_jump_matrix / sum(prev_jump_matrix)
    return new_jump_matrix


def get_ppr_reorder(image_ids, feedbacks, prev_jump_matrix):
    k = 5
    outgoing_img_graph, image_ids = create_similarity_graph(k, "euclidean", filter_string={"imageName": {"$in": image_ids}})
    new_jump_matrix = get_modified_jump_matrix(feedbacks, image_ids, prev_jump_matrix)
    transition_matrix = get_transition_matrix(outgoing_img_graph, k)
    pagerank = compute_pagerank(transition_matrix, new_jump_matrix)
    pagerank_dict = {x: y for x, y in zip(image_ids, pagerank)}
    reordered_images = [a for a, b in sorted(pagerank_dict.items(), key=lambda x: x[1], reverse=True)]
    return reordered_images, new_jump_matrix


def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", help="Number of Images to display", type=int,  required=True)
    parser.add_argument("--query_image_id", help="Query Image ID", type=str, required=True)
    parser.add_argument("--clf", help="What classifier to use", type=str, choices=["SVM", "DT", "PPR", "Probablistic"], required=True)
    return parser

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()

    mongo_client = connect_to_db()
    config = list(mongo_client.mwdb_project.images_index.find({'_id' : "config"}))[0]

    with open('index.json', 'r') as fp:
        idx_structure = json.load(fp)
    points_dict = list(mongo_client.mwdb_project.images_index.find({'_id': "points"}, {"_id" : 0}))[0]

    w_length = list(mongo_client.mwdb_project.images_index.find({"_id" : "w_length"}))[0]["w_length"]

    lsh_points = {}

    for k, v in points_dict.items():
        lsh_points[int(k)] = v

    print(config)
    query_image_details = list(mongo_client.mwdb_project.image_features.find({"imageName": args.query_image_id.split("/")[-1]}))[0] #a change was made here cuz my database does not contain imageName as a whole path

    query_vector = query_image_details["HOG_reduced"]

    imglsh = ImageLSH(config["k"], config["L"])
    imglsh.load_index_structure(idx_structure, lsh_points, w_length)

    #full_data_matrix, image_ids = get_data_matrix("HOG")
    #full_data_matrix = reduce_dimensions_lda(full_data_matrix, 256)
    # print(full_data_matrix.shape)

    similarity_scores, total_images_considered, unique_images_considered  = imglsh.find_similar_images(query_vector, args.t, mongo_client)
    jmp_matrix = None
    while True:
        pprint(similarity_scores)
        similar_images = []
        for score in similarity_scores:
            image = {}
            image["imageName"] = score[0]
            image["image_path"] = list(mongo_client.mwdb_project.image_features.find({"imageName" : score[0]}))[0]["image_path"]
            similar_images.append(image)

        plot_results(similar_images, query_image_details["image_path"]) #work has to be done here, we need R and IR to be created by this
        R, IR = get_feedback_ids()
        if args.clf == "SVM":
            similarity_scores = classify_with_svm(query_vector, similarity_scores, R, IR)
        elif args.clf == "PPR":
            similarity_scores, jmp_matrix = classify_with_ppr(query_vector, similarity_scores, R, IR)
        elif args.clf == "DT":
            similarity_scores = classify_with_dt(query_vector, similarity_scores, R, IR)
        else:
            similarity_scores = classify_with_probablistic_feedback(query_vector, similarity_scores, R)

def revise_query_vector(query_vector, R, mongo_client):
    R, _ = R_IR_image_corpus(R, mongo_client)
    return 0.55 * np.array(query_vector) + 0.45/ len(R) * np.sum(R,axis=0)


def classify_with_ppr(query_vector, similarity_scores, R, IR, prev_jump_matrix=None):
    folder_image_ids = [ val[0] for val in similarity_scores ]
    if prev_jump_matrix == None:
        prev_jump_matrix = np.ones(len(folder_image_ids)) / len(folder_image_ids)

    user_feedback = []
    for v in R:
        user_feedback.append((v, "R"))
    for v in IR:
        user_feedback.append((v, "I"))

    # feedback reordering using ppr
    reordered_images, new_jump_matrix = get_ppr_reorder(folder_image_ids, user_feedback, prev_jump_matrix)
    reordered_images = [ [v] for v in reordered_images ]
    return reordered_images, new_jump_matrix

def get_vectors_for_images(images):
    mongo_client = connect_to_db()
    result = []
    for image in images:
        result.append(np.array(list(mongo_client.mwdb_project.image_features.find({'imageName' : image}))[0]["HOG_reduced"]))
    mongo_client.close()
    return np.array(result)

def classify_with_svm(query_vector, similarity_scores, R, IR):
    train_data = np.append(get_vectors_for_images(R), get_vectors_for_images(IR), axis = 0)
    R_labels = np.ones(len(R))
    IR_labels = np.zeros(len(IR))
    training_labels = np.append(R_labels, IR_labels)

    clf = SupportVectorMachine(kernel=rbf_kernel, power=4, coef=1)
    training_labels[training_labels == 0] = -1
    clf.fit(train_data, training_labels)

    test_data = get_vectors_for_images([val[0] for val in similarity_scores])

    values = clf.predict(test_data)
    results = []
    for val, sim_score in zip(values, similarity_scores):
        if val == -1:
            results.append((sim_score[0], sim_score[1] + 1))
        else:
            results.append((sim_score[0], sim_score[1] - 1))
    return sorted(results, key=lambda tup: tup[1])

def classify_with_dt(query_vector, similarity_scores, R, IR):
    train_data = np.append(get_vectors_for_images(R), get_vectors_for_images(IR), axis = 0)
    R_labels = np.ones(len(R))
    IR_labels = np.zeros(len(IR))
    training_labels = np.reshape(np.append(R_labels, IR_labels, axis = 0), (len(R) + len(IR), 1))
    test_data = get_vectors_for_images([val[0] for val in similarity_scores])
    labeled_data = np.append(train_data, training_labels, axis = 1)
    model = DecisionTreeClassifier()
    model.fit(labeled_data)

    values = model.transform(test_data)
    results = []
    for val, sim_score in zip(values, similarity_scores):
        if val == 0:
            results.append((sim_score[0], sim_score[1] + 1))
        else:
            results.append((sim_score[0], sim_score[1] - 1))
    return sorted(results, key=lambda tup: tup[1])

def get_feedback_ids():
    R = input("Enter Relevant image IDs, comma seperated/")
    print("Relevant Images are: ", R)
    IR = input("Enter IRRelevant image IDs, comma seperated/")
    print("IRRelevant Images are: ", IR)
    return R.split(","), IR.split(",")

def convert_to_binary(data):
    mean = np.mean(data)
    result = []
    for val in data:
        if val >= mean:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

def count_occurances(data, query_vector):
    bin_data = convert_to_binary(data)
    bin_q_data = convert_to_binary(query_vector)
    count = 0
    for v, q_val in zip(bin_data, bin_q_data):
        if v == q_val:
            count = count + 1
    return count

def count_relevant_occurances(relevant_images, image_vector):
    mongo_client = connect_to_db()
    count = 0
    for img in relevant_images:
        vector = list(mongo_client.mwdb_project.image_features.find({'imageName' : img}))[0]["HOG_reduced"]
        count += count_occurances(vector, image_vector)
    mongo_client.close()
    return count

def convert_to_dict(similar_images):
    similarity_dict = {}
    for img in similar_images:
        similarity_dict[img[0]] = img[1]
    return similarity_dict

def classify_with_probablistic_feedback(query_vector, similar_images, relevant_images):
    mongo_client = connect_to_db()
    value = 0
    N = len(similar_images) * len(similar_images[0])
    R = len(relevant_images) * len(relevant_images[0])
    similarity_dict = convert_to_dict(similar_images)
    reordered_images = []
    for img in relevant_images:
        di = similarity_dict[img]
        image_vector = list(mongo_client.mwdb_project.image_features.find({'imageName': img}))[0]["HOG_reduced"]
        ni = count_occurances(image_vector, query_vector)
        ri = count_relevant_occurances(relevant_images, image_vector)
        pi = (ri + (ni / N )) / (R + 1)
        ui = (ni - ri + (ni / N)) / (N - R + 1)
        value -=  di * np.log(pi *  ( 1 - ui ) /  ui * ( 1 - pi ))
        reordered_images.append((img, value))
        similarity_dict.pop(img)
    mongo_client.close()
    for k, v in similarity_dict.items():
        reordered_images.append((k, v))

    return sorted(reordered_images, key=lambda tup: tup[1])

if __name__ == '__main__':
    main()

