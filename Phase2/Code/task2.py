#! /usr/bin/env python3
import argparse
from project_utils import *
from dimensionality_reduction import *
from feature_extraction import *
from task1 import populate_database
from distance_measures import find_similar_images

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--model", help="Name of the model to use - CM, LBP, HOG, SIFT", type=str, choices=["CM", "LBP", "HOG", "SIFT"], required=True)
    parser.add_argument("--lsa_model", help="Name of the latent semantic analysis model to use - PCA, SVD, NMF or LDA", type=str, choices=["PCA", "LDA", "NMF", "SVD"], required=True)
    parser.add_argument("--k_features", help="Number of k latent semantics to use", type=int, required=True)
    parser.add_argument("--m", help="Number of similar images to retrieve", type=int, required=True)
    parser.add_argument("--query_image", help="Query image to find similar images for - add the .jpg", type=str, required=True)
    return parser

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    populate_database(args)
    data_matrix, image_ids = get_data_matrix(args.model)
    dimensionality_reduction(data_matrix, image_ids, args, viz=True)

    similar_images = find_similar_images(args.query_image, args.model, args.m)
    for similar_image in similar_images:
        print(similar_image)
    plot_results(similar_images, args.image_folder + "/" + args.query_image)


if __name__ == "__main__" :
    main()
