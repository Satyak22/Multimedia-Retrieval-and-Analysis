#! /usr/bin/env python3
import argparse
import numpy as np
from task1 import populate_database
from dimensionality_reduction import reduce_dimensions_svd
from classifiers import DecisionTreeClassifier, SupportVectorMachine, rbf_kernel
from project_utils import get_data_matrix, convert_label_to_filterstring, enumerate_files_in_dir, connect_to_db, get_actual_labels_from_csv, get_accuracy
from sklearn.model_selection import train_test_split
from feature_extraction import extract_hog_features, extract_color_moments
from task3 import *

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", help="Path to the folder containing training images", type=str,  required=True)
    parser.add_argument("--train_metadata", help="Path to the training metadata", type=str, required=True)
    parser.add_argument("--test_folder", help="Path to the folder containing the test images", type=str, required=True)
    parser.add_argument("--classifier", help="Select classifier - SVM, DT or PPR", type=str, choices=["SVM", "DT", "PPR"], required=True)
    parser.add_argument("--labels_csv", help="Path to the original CSV", type=str)
    return parser

def label_images(dorsal_pagerank_dict, palmar_pagerank_dict, unlabeled_image_ids):
    predictions = []
    for image in unlabeled_image_ids:
        if dorsal_pagerank_dict[image] > palmar_pagerank_dict[image]:
            predictions.append((image, "dorsal"))
        elif dorsal_pagerank_dict[image] < palmar_pagerank_dict[image]:
            predictions.append((image, "palmar"))
        else:
            predictions.append((image, "unknown"))
    return predictions

def get_seed_matrix(label):
    mongo_client = connect_to_db()
    images = mongo_client.mwdb_project.image_features.find({},{"imageName":1, "_id": 0, "aspectOfHand": 1})
    seed_list = []
    count = 0
    for img in images:
        if "aspectOfHand" in img:
            if label in img["aspectOfHand"].lower():
                count += 1
                seed_list.append(1)
            else:
                seed_list.append(0)
        else:
            seed_list.append(0)
    seed_matrix = np.array(seed_list)
    seed_matrix = seed_matrix/np.sum(seed_matrix)
    return seed_matrix


def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    populate_database(args)
    model = "CM"

    dorsal_data_matrix, _ = get_data_matrix(model, convert_label_to_filterstring("dorsal"))

    palmar_data_matrix, _ = get_data_matrix(model, convert_label_to_filterstring("palmar"))

    dorsal_labels = np.zeros((dorsal_data_matrix.shape[0], 1))
    palmar_labels = np.ones((palmar_data_matrix.shape[0], 1))

    labels = np.append(dorsal_labels, palmar_labels, axis = 0)

    combined_data = np.append(dorsal_data_matrix, palmar_data_matrix, axis = 0)
    #reduced_data = reduce_dimensions_svd(combined_data, 20)

    reduced_data, v_matrix  = reduce_dimensions_svd(combined_data, 20, get_v = True)
    dx, ddx, labels, d_labels = train_test_split(reduced_data, labels, test_size = 0.1, random_state = 42)

    reduced_data = np.append(dx, ddx, axis = 0)
    labels = np.append(labels, d_labels, axis = 0)

    labeled_data = np.append(reduced_data, labels, axis = 1)

    testing_images, test_image_ids = enumerate_files_in_dir(args.test_folder)

    test_dataset = []
    for test_image, image_id in zip(testing_images, test_image_ids):
        #test_dataset.append(np.array(extract_hog_features(test_image)))
        test_dataset.append(np.array(extract_color_moments(test_image)))

    test_dataset = np.array(test_dataset)

    #reduced_test_dataset = reduce_dimensions_svd(test_dataset, 20)
    reduced_test_dataset = np.matmul(test_dataset, v_matrix)

    mongo_client = connect_to_db()
    actual_labels = get_actual_labels_from_csv(args.labels_csv, test_image_ids)
    predicted = []

    if args.classifier == "DT":
        model = DecisionTreeClassifier()
        model.fit(labeled_data)
        results = model.transform(reduced_test_dataset)
        for test_image_id, result in zip(test_image_ids, results):
            if result == 0:
                label = "dorsal"
            elif result == 1:
                label = "palmar"
            predicted.append((test_image_id, label))
            print("{0} - {1}".format(test_image_id, label))

    elif args.classifier == "SVM":
        clf = SupportVectorMachine(kernel=rbf_kernel, power=4, coef=1)
        training_labels = labels[:]
        # SVM needs labels to be 1, and -1
        training_labels[training_labels == 0] = -1
        clf.fit(reduced_data, training_labels)
        values = clf.predict(reduced_test_dataset)
        print(values)
        for test_image_id, result in zip(test_image_ids, values):
            if result == 1:
                label = "palmar"
            else:
                label = "dorsal"
            predicted.append((test_image_id, label))
            print("{0} - {1}".format(test_image_id, label))

    elif args.classifier == "PPR":
        args.k = 15
        function_val = "manhattan"

        #process_all_images(args.train_folder, "CM")
        #process_all_images(args.test_folder, "CM")
        outgoing_img_graph, image_ids = create_similarity_graph(args.k, function_val, "CM")
        transition_matrix = get_transition_matrix(outgoing_img_graph, args.k)

        seed_matrix_dorsal = get_seed_matrix("dorsal")
        seed_matrix_palmar = get_seed_matrix("palmar")

        dorsal_pagerank = compute_pagerank(transition_matrix, seed_matrix_dorsal)
        palmar_pagerank = compute_pagerank(transition_matrix, seed_matrix_palmar)
        dorsal_pagerank_dict = {x:y for x,y in zip(image_ids, dorsal_pagerank)}
        palmar_pagerank_dict = {x:y for x,y in zip(image_ids, palmar_pagerank)}

        predicted = label_images(dorsal_pagerank_dict, palmar_pagerank_dict, test_image_ids)

    print(get_accuracy(actual_labels, predicted))
    mongo_client.close()

if __name__ == '__main__':
    main()
