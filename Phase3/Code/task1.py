#! /usr/bin/env python3
import argparse
from project_utils import *
from dimensionality_reduction import *
from feature_extraction import *
import pandas as pd

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", help="Path to the folder containing training images", type=str,  required=True)
    parser.add_argument("--train_metadata", help="Path to the training metadata", type=str, required=True)
    parser.add_argument("--test_folder", help="Path to the folder containing the test images", type=str, required=True)
    parser.add_argument("--k", help="Value of K", type=int, required=True)
    parser.add_argument("--labels_csv", help="Path to the CSV file that contains the labels for unlabelled images, used for accuracy calculation", type=str)
    return parser

# Driver Method that parallelly extracts the features of all images in a directory, based on the specified model
def process_all_images(image_path):
    full_paths, image_ids = enumerate_files_in_dir(image_path)
    print("There are {0} images in the directory".format(len(full_paths)))
    mongo_client = connect_to_db()
    images_in_db_dict = list(mongo_client.mwdb_project.image_features.find({}, {"imageName":1, "_id" : 0, "HOG" : 1}))
    mongo_client.close()
    images_in_db = []
    for image in images_in_db_dict:
        if "CM" in image:
            images_in_db.append(image["imageName"])

    if sorted(images_in_db) == sorted(image_ids):
        print("Db already contains features for CM")
        return

    pool = Pool()
    pool.map(extract_hog_features, full_paths)
    pool.map(extract_color_moments, full_paths)
    pool.close()
    pool.join()
    return

def process_metadata(args):
    print("Processing Metadata...")
    _, image_ids = enumerate_files_in_dir(args.train_folder)
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    images_in_db_dict = list(mongo_client.mwdb_project.image_features.find({}, {"imageName":1, "_id" : 0}))
    images_in_db = []
    for image in images_in_db_dict:
        images_in_db.append(image["imageName"])
    if sorted(images_in_db) == sorted(image_ids):
        print("Db already populated, skipping")
        return
    full_dataset = pd.read_csv(args.train_metadata)
    new_df = full_dataset.loc[full_dataset['imageName'].isin(image_ids),:]
    values = new_df.T.to_dict()
    # Clear the collection before inserting new ones
    db_handle.image_features.drop()
    for key, value in values.items():
        db_handle.image_features.insert_one(value)
    mongo_client.close()

def populate_database(args):
    process_metadata(args)
    print("Extracting CM of images in {0}".format(args.train_folder))
    process_all_images(args.train_folder)
    print("Extraction complete, please inspect database.")

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    populate_database(args)
    mongo_client = connect_to_db()
    testing_images, test_image_ids = enumerate_files_in_dir(args.test_folder)

    model = "CM"
    dorsal_data_matrix, _ = get_data_matrix(model, convert_label_to_filterstring("dorsal"))
    palmar_data_matrix, _ = get_data_matrix(model, convert_label_to_filterstring("palmar"))

    _, dorsal_latent_features = reduce_dimensions_svd(dorsal_data_matrix, args.k, get_v=True)
    _, palmar_latent_features = reduce_dimensions_svd(palmar_data_matrix, args.k, get_v=True)

    predicted = []
    for test_image, image_id in zip(testing_images, test_image_ids):
        tfv = extract_color_moments(test_image)
        dorsal_score = np.mean(np.matmul(dorsal_latent_features.T, tfv))
        palmar_score = np.mean(np.matmul(palmar_latent_features.T, tfv))
        if dorsal_score < palmar_score:
            label = "dorsal"
        else:
            label = "palmar"
        predicted.append((image_id, label))
    print(predicted)

    if "labels_csv" in args:
        actuals = get_actual_labels_from_csv(args.labels_csv, testing_images)
        accuracy = get_accuracy(actuals, predicted)
        print("accuracy is {0}".format(accuracy))

if __name__ == "__main__":
    main()
