#! /usr/bin/env python3
import argparse
from project_utils import *
import pprint
import numpy as np
from collections import Counter
def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--model", help="Name of the model to use - CM, LBP, HOG, SIFT", type=str, choices=["CM", "LBP", "HOG", "SIFT"], required=True)
    parser.add_argument("--lsa_model", help="Name of the latent semantic analysis model to use - PCA, SVD, NMF or LDA", type=str, choices=["PCA", "LDA", "NMF", "SVD"], required=True)
    parser.add_argument("--k_features", help="Number of k latent semantics to use", type=int, required=True)
    parser.add_argument("--label", help="Labels for the image", type=str, required=True, choices=["left-hand", "right-hand", "dorsal", "palmar", "with accessories", "without accessories", "male", "female"])
    return parser

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    filterstring = convert_label_to_filterstring(args.label)
    mongo_client = connect_to_db()
    images = list(mongo_client.mwdb_project.image_features.find(filterstring, {"imageName" : 1}))
    mongo_client.close()
    print("There are {0} images in the db matching the label - {1}".format(len(images), args.label))

    data_matrix, image_ids = get_data_matrix(args.model, filterstring)

    dimensionality_reduction(data_matrix, image_ids, args, args.label, viz=True)

if __name__ == "__main__" :
    main()
