#! /usr/bin/env python3
import argparse
from project_utils import *
from feature_extraction import *
from distance_measures import find_similar_images
from matplotlib import pyplot as plt
import csv
from task1 import process_all_images

def experiment():
    features = ["LBP", "HOG", "SIFT"]
    reducers = ["SVD","PCA","NMF","LDA"]
    for k in range(15, 25):
        for feature in features:
            for reducer in reducers:
                anythings = main(feature,reducer,k) # the similarity score check
                for key in anythings:
                    subject, score= key,anythings[key]
                    list1 = [feature, reducer, k, score, subject]
                    with open("score_task6.csv", "a") as fp:
                        wr = csv.writer(fp, dialect='excel')
                        wr.writerow(list1)

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file", type=str,  required=True)
    parser.add_argument("--model", help="Name of the model to use - CM, LBP, HOG, SIFT", type=str, choices=["CM", "LBP", "HOG", "SIFT"], required=True)
    parser.add_argument("--lsa_model", help="Name of the latent semantic analysis model to use - PCA, SVD, NMF or LDA", type=str, choices=["PCA", "LDA", "NMF", "SVD"], required=True)
    parser.add_argument("--subjectId", help="Id of the subject to search for.", type=int, required=True)
    parser.add_argument("--k_features", help="Value of K", type=int)
    return parser

def find_distance_between_query_and_subject(args, query_feature_vector, subject_feature_vectors, subject_ids):
    data_matrix = np.insert(subject_feature_vectors, 0, query_feature_vector, axis = 0)
    print(data_matrix.shape)
    subject_ids.insert(0, -1)
    print("Reducing dimensions...")
    reduced_data_matrix, _ = dimensionality_reduction(data_matrix, [], args, label=str(args.subjectId))
    reduced_query_fv = reduced_data_matrix[0]
    subject_similarity_scores = {}
    print("Finding similarity...")
    for i in range(1, len(reduced_data_matrix)):
        subject_fv = reduced_data_matrix[i]
        distance_score = 0
        for q, s in zip(reduced_query_fv, subject_fv):
            distance_score += np.square(q - s)
        distance_score = np.sqrt(distance_score)
        if distance_score == 0:
            subject_similarity_scores[str(subject_ids[i])] = 100
        else:
            subject_similarity_scores[str(subject_ids[i])] = (1 / distance_score) * 100
    return subject_similarity_scores

def finalviz(query_subject_id, subject_similarity_score):
    other_subject_image_names = []
    mongo_client =  connect_to_db()
    db_handle = mongo_client.mwdb_project
    query_subject_images = list(db_handle.image_features.find({'id' : query_subject_id}, {'imageName': 1, '_id': 0, 'image_path':1}))
    sorted_sub_similarity_score = sorted(subject_similarity_score.items(), key=lambda x: x[1])
    sorted_sub_similarity_score = sorted_sub_similarity_score[0:3]
    subject_ids = list([i[0] for i in sorted_sub_similarity_score])
    for other_subject_id in subject_ids:
        other_subject_image_names.append(list(db_handle.image_features.find({'id': int(other_subject_id)}, {'imageName': 1, '_id': 0, 'image_path': 1, 'id':1})))
    rows,i,j= 4,0,0
    cols = len(query_subject_images)
    for other in other_subject_image_names:
        if cols < len(other):
            cols = len(other)
    f, s_arr = plt.subplots(rows, cols)
    for h in range(0,rows):
        for k in range(0,cols):
            s_arr[h][k].axis("off")

    s_arr[0][0].text(0.5,-0.1, "Target Image", size=6, ha="center", transform=s_arr[0][0].transAxes)
    for query_image in query_subject_images:
        s_arr[0][j].imshow(plt.imread(query_image["image_path"]))
        j = j+1
    for other_images in other_subject_image_names:
        i = i+1
        j = 0
        for image in other_images:
            if j == 0:
                s_arr[i][j].text(0.5, -0.1,"Subject Id: " + str(image["id"]) + ", similarity score: " + str(subject_similarity_score[str(image["id"])]), size=6, ha="center", transform=s_arr[i][j].transAxes)
            s_arr[i][j].imshow(plt.imread(image["image_path"]))
            j = j+1
    plt.suptitle(fname)
    plt.show()
    mongo_client.close()
    plt.close()

def get_average_features(feature_vector):
    return np.mean(feature_vector, axis=0)

#def main(model,lsa,k):
def main():
    parser = setup_arg_parse()
    args = parser.parse_args()
    args.model = "HOG"
    args.lsa_model = "LDA"
    args.k_features = 19
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    query_subject_images = list(db_handle.image_features.find({'id' : args.subjectId}, {'imageName' : 1, '_id' : 0}))
    print("Image belonging to the subject are", query_subject_images)

    if len(query_subject_images) == 0:
        print("No images for the given subject in the dataset.")
        subject_image_path = input("Enter path to the subject's Images: ")
        print("Processing images present in {0}".format(subject_image_path))
        process_all_images(subject_image_path, args.model)
        query_subject_images = list(db_handle.image_features.find({'id' : args.subjectId}, {'imageName' : 1, '_id' : 0}))

    query_feature_vec = []
    for query_subject in query_subject_images:
        img_fv = list(db_handle.image_features.find({'imageName' : query_subject['imageName']}, {args.model: 1, '_id' : 0}))
        for ifv in img_fv:
            if args.model == "SIFT":
                avg_sift_features = get_average_features(np.array(ifv[args.model]))
                query_feature_vec.append(avg_sift_features)
            else:
                query_feature_vec.append(np.array(ifv[args.model]))

    query_fv = get_average_features(np.array(query_feature_vec))

    # first let us see how many unique subject Ids are present
    other_subjects = list(db_handle.image_features.find({'id': {"$ne" : args.subjectId}}, {'id' : 1, '_id' : 0}).distinct('id'))
    subject_similarity_score = {}
    subject_features = []
    for subject in other_subjects:
        print("Processing distance between {0} and {1}".format(args.subjectId, subject))
        subject_image_dict = list(db_handle.image_features.find({'id' : subject}, {args.model : 1,'_id' : 0}))
        subject_fv = []
        for img in subject_image_dict:
            if args.model == "SIFT":
                avg_sift_features = get_average_features(np.array(img[args.model]))
                subject_fv.append(avg_sift_features)
            else:
                subject_fv.append(img[args.model])
        avg_sfv = get_average_features(np.array(subject_fv))
        subject_features.append(avg_sfv)

    subject_similarity_score = find_distance_between_query_and_subject(args, query_fv, np.array(subject_features), other_subjects)
    finalviz(query_subject_id=args.subjectId, subject_similarity_score = subject_similarity_score)
    mongo_client.close()
    print({x:y for x,y in sorted(subject_similarity_score.items(), key = lambda x:x[1], reverse = True)[:3]})

if __name__ == "__main__" :
    main()
