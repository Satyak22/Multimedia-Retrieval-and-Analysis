#! /usr/bin/env python3
import argparse
from project_utils import *
from sklearn.decomposition import NMF

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--k", help="K latent semantics", type=int, required=True)
    return parser

def construct_image_metadata_matrix(path_to_metadata_file):

    full_dataset = pd.read_csv(path_to_metadata_file)
    new = full_dataset["aspectOfHand"].str.split(" ", n = 1, expand = True)

    full_dataset["side"] = new[0]
    full_dataset["leftORright"] = new[1]
    full_dataset = pd.concat([full_dataset,pd.get_dummies(full_dataset['gender'], prefix='gender')],axis=1)
    full_dataset = pd.concat([full_dataset,pd.get_dummies(full_dataset['accessories'], prefix='accessories')],axis=1)
    full_dataset = pd.concat([full_dataset,pd.get_dummies(full_dataset['side'], prefix='side')],axis=1)
    full_dataset = pd.concat([full_dataset,pd.get_dummies(full_dataset['leftORright'], prefix='leftORright')],axis=1)

    full_dataset.drop(columns=["aspectOfHand", "gender", "accessories", "side", "leftORright","age", "id", "irregularities", "nailPolish", "skinColor"], inplace=True)
    # print(full_dataset)

    full_dataset.set_index('imageName', inplace=True)
    print(full_dataset.head())
    return full_dataset.to_numpy(), full_dataset.index.values

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    image_meta_data_matrix, images = construct_image_metadata_matrix(args.metadata_file)
    print(image_meta_data_matrix)
    reduce_dimensions_nmf(image_meta_data_matrix, args.k, images, viz=True)

if __name__ == "__main__" :
    main()
