#! /usr/bin/env python3
import argparse
from LSH import ImageLSH
from project_utils import connect_to_db, plot_results
from feature_extraction import extract_hog_features
import json

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", help="Number of Images to display", type=int,  required=True)
    parser.add_argument("--query_image_id", help="Query Image ID", type=str, required=True)
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

    query_image_details = list(mongo_client.mwdb_project.image_features.find({"imageName": args.query_image_id}))[0]

    query_vector = query_image_details["HOG_reduced"]

    imglsh = ImageLSH(config["k"], config["L"])
    imglsh.load_index_structure(idx_structure, lsh_points, w_length)

    similarity_scores, total_images_considered, unique_images_considered  = imglsh.find_similar_images(query_vector, args.t, mongo_client)

    similar_images = []
    for score in similarity_scores:
        image = {}
        image["distance_score"] = score[1]
        image["imageName"] = score[0]
        image["image_path"] = list(mongo_client.mwdb_project.image_features.find({"imageName" : score[0]}))[0]["image_path"]
        similar_images.append(image)


    mongo_client.close()
    print("Number of unique images considered = {0}".format(total_images_considered))
    print("Number of overall Images considerd = {0}".format(unique_images_considered))

    plot_results(similar_images, query_image_details["image_path"])


if __name__ == "__main__":
    main()
