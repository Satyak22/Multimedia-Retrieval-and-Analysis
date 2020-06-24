#! /usr/bin/env python3
import argparse
from project_utils import *
from feature_extraction import *
from distance_measures import find_similar_images
from sklearn.decomposition import NMF


def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--k_features", help="K features to extract", required=True, type=int)
    return parser

def find_distance_between_query_and_subject(query_idx, subject_idx, subjectFV):
    query_fv = subjectFV[query_idx]
    subject_fv = subjectFV[subject_idx]
    distance = 0
    for q, s in zip(query_fv, subject_fv):
        distance = np.square(q-s)
    distance = np.sqrt(distance)
    if distance == 0:
        return 100
    else:
        return (1/distance) * 100

def get_avg_feature_vector_for_subject(subjectId):
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    fv = []
    for val in db_handle.image_features.find({'id' : subjectId}, {'HOG' : 1, '_id' : 0}):
        fv.append(np.array(val['HOG']))
    mongo_client.close()
    return np.mean(np.array(fv), axis = 0)

def main():
    parser = setup_arg_parse()
    args = parser.parse_args()

    #The logic is to get the average feature vector for each subject, and then do dimensionality reduction, and then do a subject subject similarity
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project

    subjects = list(db_handle.image_features.find({}, {'id' : 1, '_id' : 0}).distinct('id'))
    print("There are {0} subjects in the dataset".format(len(subjects)))
    subjectFV = []
    for subject in subjects:
        subjectFV.append(get_avg_feature_vector_for_subject(subject))
    print("Dimensionality reduction...")
    reduced_subject_fv = reduce_dimensions_lda(subjectFV, 19, [])
    #print(reduced_subject_fv.shape)

    similarity_score = [[-1 for x in range(len(subjects))] for y in range(len(subjects))]
    for query_idx in range(len(subjects)):
        for sub_idx in range(len(subjects)):
            subject = subjects[sub_idx]
            print("\n=========== For Subject pair [{}][{}] ===========".format(query_idx,sub_idx))
            if query_idx == sub_idx:
                print("Same Subjects")
                similarity_score[query_idx][sub_idx] = 100
                continue
            if similarity_score[sub_idx][query_idx] != -1:
                print("Similarity_score already present for these subjects\n")
                similarity_score[query_idx][sub_idx] = similarity_score[sub_idx][query_idx]
                continue
            print("Calculating Similarity Score....")
            similarity_score[query_idx][sub_idx] = find_distance_between_query_and_subject(query_idx, sub_idx, reduced_subject_fv)
    similarity_matrix = np.array(similarity_score)
    reduce_dimensions_nmf(similarity_matrix, args.k_features, [i for i in range(len(similarity_score))], viz=True)
    mongo_client.close()

if __name__ == "__main__" :
    main()
