#! /usr/bin/env python3
import argparse
from LSH import ImageLSH
from project_utils import connect_to_db
import json
def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", help="Number of Layers", type=int,  required=True)
    parser.add_argument("--k", help="Hashes per layer", type=int, required=True)
    return parser

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    imglsh = ImageLSH(args.L, args.k)
    reduced_data, image_ids = imglsh.load_data()


    idx_structure, points_dict = imglsh.create_index_structure()

    for key, val in idx_structure.items():
        idx_structure[key] = list(val)

    lsh_points = {}
    for key, val in points_dict.items():
        lsh_points[str(key)] = val


    idx_structure['_id'] = "index"
    lsh_points['_id'] = "points"
    mongo_client = connect_to_db()

    for img, data in zip(image_ids, reduced_data):
        mongo_client.mwdb_project.image_features.update_one({'imageName': img}, {"$set" : { "HOG_reduced": data.tolist()}})

    mongo_client.mwdb_project.images_index.insert_one(lsh_points)
    w_length = { '_id' : "w_length", "w_length" : imglsh.w_length }

    mongo_client.mwdb_project.images_index.insert_one(w_length)

    config  = { 'k' : args.k, 'L': args.L , '_id' : "config"}
    mongo_client.mwdb_project.images_index.insert_one(config)
    with open('index.json', 'w') as fp:
        json.dump(idx_structure, fp)
    mongo_client.close()

if __name__ == "__main__":
    main()
