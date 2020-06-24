#! /usr/bin/env python3
import argparse
from project_utils import *
from dimensionality_reduction import *
from feature_extraction import *
def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--model", help="Name of the model to use - CM, LBP, HOG, SIFT", type=str, choices=["CM", "LBP", "HOG", "SIFT"], required=False)
    parser.add_argument("--lsa_model", help="Name of the latent semantic analysis model to use - PCA, SVD, NMF or LDA", type=str, choices=["PCA", "LDA", "NMF", "SVD"], required=True)
    parser.add_argument("--k_features", help="Number of k latent semantics to use", type=int, required=True)
    return parser

# Driver Method that parallelly extracts the features of all images in a directory, based on the specified model
def process_all_images(image_path, model):
    full_paths, image_ids = enumerate_files_in_dir(image_path)
    print("There are {0} images in the directory".format(len(full_paths)))
    mongo_client = connect_to_db()
    images_in_db_dict = list(mongo_client.mwdb_project.image_features.find({}, {"imageName":1, "_id" : 0, model: 1}))
    mongo_client.close()
    images_in_db = []
    for image in images_in_db_dict:
        if model in image:
            images_in_db.append(image["imageName"])

    if sorted(images_in_db) == sorted(image_ids):
        print("Db already contains features for {0}".format(model))
        return

    pool = Pool()
    if model == "CM":
        pool.map(extract_color_moments, full_paths)
    elif model == "LBP":
        pool.map(extract_lbp_features, full_paths)
    elif model == "HOG":
        pool.map(extract_hog_features, full_paths)
    elif model == "SIFT":
        pool.map(extract_sift_features, full_paths)
    pool.close()
    pool.join()
    return

def process_metadata(args):
    print("Processing Metadata...")
    _, image_ids = enumerate_files_in_dir(args.image_folder)
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    images_in_db_dict = list(mongo_client.mwdb_project.image_features.find({}, {"imageName":1, "_id" : 0}))
    images_in_db = []
    for image in images_in_db_dict:
        images_in_db.append(image["imageName"])
    if sorted(images_in_db) == sorted(image_ids):
        print("Db already populated, skipping")
        return
    full_dataset = pd.read_csv(args.metadata_file)
    new_df = full_dataset.loc[full_dataset['imageName'].isin(image_ids),:]
    values = new_df.T.to_dict()
    # Clear the collection before inserting new ones
    db_handle.image_features.drop()
    for key, value in values.items():
        db_handle.image_features.insert_one(value)
    mongo_client.close()

def populate_database(args):
    process_metadata(args)
    if args.model is not None:
        print("Extracting {0} of images in {1}".format(args.model, args.image_folder))
        process_all_images(args.image_folder, args.model)
    else:
        for model in ["CM", "LBP", "HOG", "SIFT"]:
            print("Extracting {0} of images in {1}".format(model, args.image_folder))
            process_all_images(args.image_folder, model)
    print("Extraction complete, please inspect database.")

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    populate_database(args)
    data_matrix, image_ids = get_data_matrix(args.model)
    dimensionality_reduction(data_matrix, image_ids, args, viz=True)

if __name__ == "__main__":
    main()
