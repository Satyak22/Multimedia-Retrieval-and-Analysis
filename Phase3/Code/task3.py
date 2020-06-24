from feature_extraction import *
from project_utils import *
from dimensionality_reduction import *
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance

def setup_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help="Path to the folder containing images", type=str,  required=True)
    parser.add_argument("--metadata_file", help="Path to the metadata file for the selected set of images", type=str, required=True)
    parser.add_argument("--query_images", help="3 user image_ids separated by comma without space", type=str, required=True)
    parser.add_argument("--k", help="The number of outgoing edges for each node in the graph", type=int, required=True)
    parser.add_argument("--K", help="The number of dominant images to be displayed", type = int, required=True)
    return parser


def process_all_images(image_path, feature_model = "HOG"):
    full_paths, image_ids = enumerate_files_in_dir(image_path)
    print("There are {0} images in the directory".format(len(full_paths)))
    mongo_client = connect_to_db()
    images_in_db_dict = list(mongo_client.mwdb_project.image_features.find({}, {"imageName":1, "_id" : 0, feature_model : 1}))
    mongo_client.close()
    images_in_db = []
    for image in images_in_db_dict:
        if feature_model in image:
            images_in_db.append(image["imageName"])

    if sorted(images_in_db) == sorted(image_ids):
        print("Db already contains features for " + feature_model)
        return

    pool = Pool()
    if feature_model == "CM":
    	pool.map(extract_color_moments, full_paths)
    elif feature_model == "HOG":
    	pool.map(extract_hog_features, full_paths)
    elif feature_model == "SIFT":
    	pool.map(extract_sift_features, full_paths)
    else:
    	pool.map(extract_lbp_features, full_paths)
    pool.close()
    pool.join()
    return

def process_metadata(path, metadata_path):
    print("Processing Metadata...")
    _, image_ids = enumerate_files_in_dir(path)
    mongo_client = connect_to_db()
    db_handle = mongo_client.mwdb_project
    images_in_db_dict = list(mongo_client.mwdb_project.image_features.find({}, {"imageName":1, "_id" : 0}))
    images_in_db = []
    for image in images_in_db_dict:
        images_in_db.append(image["imageName"])
    if sorted(images_in_db) == sorted(image_ids):
        print("Db already populated, skipping")
        return
    full_dataset = pd.read_csv(metadata_path)
    new_df = full_dataset.loc[full_dataset['imageName'].isin(image_ids),:]
    values = new_df.T.to_dict()
    # Clear the collection before inserting new ones
    db_handle.image_features.drop()
    for key, value in values.items():
        db_handle.image_features.insert_one(value)
    mongo_client.close()


def find_distance_score(curr_img, data_matrix, function_val):
	
	##################### Minkowski 3 #######################
	
	if function_val == "minkowski3":
		dis_score = []
		for img in data_matrix:
			dis_score.append(distance.minkowski(img, curr_img, 3))

		return dis_score


	################ Cosine #######################
	elif function_val == "cosine":
		sim_score = []
		for img in data_matrix:
			dot = np.dot(curr_img, img)
			norm1 = np.linalg.norm(curr_img)
			norm2 = np.linalg.norm(img)
			sim_score.append(dot/(norm1 * norm2))
		return sim_score


	############## Euclidean Distance ####################
	elif function_val == "euclidean":
		return np.sqrt(np.sum(np.square(data_matrix - curr_img),axis = 1))

	################# Manhattan Distance ###################
	else:
		return np.sum(np.absolute(data_matrix - curr_img), axis = 1)


def create_similarity_graph(k, function_val, feature_model = "HOG", filter_string=None):
	if filter_string:
		data_matrix, image_ids = get_data_matrix(feature_model, filterstring=filter_string)
	else:
		data_matrix, image_ids = get_data_matrix(feature_model)
	img_graph = {}
	for idx in range(data_matrix.shape[0]):
		curr_img = data_matrix[idx]
		curr_img_id = image_ids[idx]
		distance_vector = find_distance_score(curr_img, data_matrix, function_val)
		img_distance_map = {x:y for x,y in zip(range(len(image_ids)),distance_vector)}
		if function_val == "cosine":
			img_distance_tuple = sorted(img_distance_map.items(), key = lambda x: x[1], reverse = True)[:k]
		else:
			img_distance_tuple = sorted(img_distance_map.items(), key = lambda x: x[1])[:k]
		img_graph[idx] = [x for x,y in img_distance_tuple]
	return img_graph, image_ids

def get_transition_matrix(outgoing_img_graph, k):
	transition_matrix = [[0 for x in range(len(outgoing_img_graph))] for y in range(len(outgoing_img_graph))]
	for key in outgoing_img_graph:
		for val in outgoing_img_graph[key]:
			transition_matrix[key][val] = 1/k
	return np.array(transition_matrix).T

def get_jump_matrix(user_nodes, image_ids):
	jump_matrix = [0 for x in range(len(image_ids))]
	for x in user_nodes:
		jump_matrix[image_ids.index(x)] = 1/len(user_nodes)
	return np.array(jump_matrix)


def compute_pagerank(transition_matrix, jump_matrix):
	total_img_count = transition_matrix.shape[0]
	beta = 0.85

	new_pagerank = np.array([1/total_img_count for x in range(total_img_count)])
	old_pagerank = np.array([1/total_img_count for x in range(total_img_count)])


	for x in range(60):
		new_pagerank = beta*np.matmul(transition_matrix,old_pagerank.T) + (1-beta)*jump_matrix.T
		old_pagerank = new_pagerank.T

	return new_pagerank

def visualize(images, path):
	rows = int(np.sqrt(len(images))) + 1
	cols = int(np.ceil(len(images)/rows))
	fig, s_arr = plt.subplots(rows,cols)
	i = j = 0
	for x in range(len(images)):
		s_arr[i,j].axis("off")
		s_arr[i,j].text(0.5,-0.1, images[x], size=6, ha="center", transform=s_arr[i,j].transAxes)
		s_arr[i,j].imshow(plt.imread(path + "/" + images[x]))
		j+=1

		if j >= cols:
			j = 0
			i+=1

	while i<rows:
		while j<cols:
			s_arr[i][j].axis("off")
			j+=1
		i += 1
		j = 0

	plt.show()


def main():
	parser = setup_arg_parse()
	args = parser.parse_args()
	user_nodes = args.query_images.split(",")
	process_metadata(args.image_folder, args.metadata_file)
	process_all_images(args.image_folder)
	outgoing_img_graph, image_ids = create_similarity_graph(args.k, "euclidean")
	jump_matrix = get_jump_matrix(user_nodes, image_ids)
	transition_matrix = get_transition_matrix(outgoing_img_graph, args.k)
	pagerank = compute_pagerank(transition_matrix, jump_matrix)
	pagerank_dict = {x:y for x,y in zip(image_ids,pagerank)}
	K_dominant_images = [a for a,b in sorted(pagerank_dict.items(), key = lambda x:x[1], reverse = True)[:args.K]]
	visualize(K_dominant_images, args.image_folder)


if __name__ == "__main__":
	main()
