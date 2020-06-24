import json
import math
import os
import random
import numpy
import pandas as pd
from scipy.spatial import distance
from project_utils import get_data_matrix
from dimensionality_reduction import reduce_dimensions_lda, reduce_dimensions_svd

class ImageLSH():
    def __init__(self, num_layers, num_hashs):
        print("Initializing LSH index with {0} Layers and {1} Hashes".format(num_layers, num_hashs))
        self.num_layers = num_layers
        self.num_hashs = num_hashs
        self.latent_range_dict = {}
        self.lsh_points_dict = {}
        self.lsh_range_dict = {}
        self.image_bucket_df = pd.DataFrame()
        self.image_latent_df = pd.DataFrame()
        self.w_length = 0.0

    def load_data(self):
        self.data_matrix, self.image_ids = get_data_matrix("HOG_reduced")
        self.reduced_data = self.data_matrix
        return self.reduced_data, self.image_ids

    def assign_group(self, value):
        """
        Assigns bucket
        :param value:
        :return: bucket
        """
        if value < 0:
            return math.floor(value/self.w_length)
        else:
            return math.ceil(value / self.w_length)

    def init_lsh_vectors(self, U_dataframe):
        """
        initialize lsh vectors
        :param U_dataframe:
        """
        # First finds range for each column in the df
        # Them, finds uniform distributions of the range for each column
        # That is assigned to each lsh point
        # The lsh distance from 0 vector to the lsh points list is found
        print("Initializing the LSH vectors")
        origin = list(numpy.zeros(shape=(1, 256)))
        for column in U_dataframe:
            self.latent_range_dict[column] = (U_dataframe[column].min(), U_dataframe[column].max())

        for i in range(0, self.num_layers * self.num_hashs):
            cur_vector_list = []
            for column in U_dataframe:
                cur_vector_list.append(random.uniform(self.latent_range_dict[column][0], self.latent_range_dict[column][1]))
            self.lsh_points_dict[i] = cur_vector_list
            self.lsh_range_dict[i] = distance.euclidean(origin, cur_vector_list)

    def project_on_hash_function(self, image_vector, lsh_vector):
        """
        projection of image vector on the hash fn
        :param image_vector:
        :param lsh_vector:
        :return: projection value
        """
        image_lsh_dot_product = numpy.dot(image_vector, lsh_vector)
        if image_lsh_dot_product == 0.0:
            return 0
        lsh_vector_dot_product = numpy.dot(lsh_vector, lsh_vector)
        projection = image_lsh_dot_product/lsh_vector_dot_product*lsh_vector
        projection_magnitude = numpy.linalg.norm(projection)
        return projection_magnitude


    def LSH(self, vector):
        """
        list of buckets for the vector
        :param vector:
        :return:
        """
        bucket_list = []
        for lsh_vector in range(0, len(self.lsh_points_dict)):
            bucket_list.append(self.assign_group(self.project_on_hash_function(numpy.array(vector), numpy.array(self.lsh_points_dict[lsh_vector]))))
        return bucket_list

    def group_data(self):
        """
        groups all images into buckets
        :return:
        """
        print("Grouping data into buckets")
        reduced_df = pd.DataFrame(self.reduced_data)
        self.init_lsh_vectors(reduced_df)
        self.w_length = min(self.lsh_range_dict.values()) / float(100)

        bucket_matrix = numpy.zeros(shape=(len(self.reduced_data), len(self.lsh_points_dict)))
        # the shape is number of samples in U * (L * k)

        for image in range(0, len(self.reduced_data)):
            bucket_matrix[image] = self.LSH(self.reduced_data[image])


        image_id_df = pd.DataFrame(self.image_ids, columns=['image_id'])
        self.image_latent_df = reduced_df.join(image_id_df, how="left")
        return pd.DataFrame(bucket_matrix).join(image_id_df, how="left")

    def index_data(self):
        """
        Assigns buckets to images in the dataframe
        :param df:
        :return:
        """
        print("Indexing the structure..")
        index_structure_dict = {}
        counterval = 0
        for index, row in self.image_bucket_df.iterrows():
            image_id = row["image_id"]
            column = 0
            for i in range(0, self.num_layers):
                bucket = ""
                for j in range(0, self.num_hashs):
                    interval = row[column]
                    bucket = bucket + str(int(interval)) + "-"
                    column += 1
                    if bucket.strip("-") in index_structure_dict:
                        index_structure_dict[bucket.strip("-")].add(image_id)
                    else:
                        image_set = set()
                        image_set.add(image_id)
                        index_structure_dict[bucket.strip("-")] = image_set

        return index_structure_dict

    def fetch_hash_keys(self, bucket_list):
        """
        Obtain the hash keys for the bucket list
        :param bucket_list:
        :return:
        """
        column = 0
        hash_key_list = []
        for i in range(0, self.num_layers):
            bucket = ""
            for j in range(0, self.num_hashs):
                interval = bucket_list[column]
                if(j != self.num_hashs - 1):
                    bucket = bucket + str(int(interval)) + "-"
                else:
                    bucket = bucket + str(int(interval))
                column += 1
            hash_key_list.append(bucket)
        return hash_key_list

    def create_index_structure(self):
        """
        Creates index structure for search
        :param image_list:
        """
        # this contains the bucketed data and the image id on the right most column
        self.image_bucket_df = self.group_data()
        self.index_structure = self.index_data()
        return self.index_structure, self.lsh_points_dict

    def load_index_structure(self, idx, points_dict, w_length):
        self.index_structure = idx
        self.lsh_points_dict = points_dict
        self.w_length = w_length

    def find_similar_images(self, query_vector, no_of_images, mongo_client):
        """
        Nearest neighbor for the query vector
        :param query_vector:
        :param no_of_nearest_neighbours:
        :return: list of r nearest images
        """
        query_bucket_list = self.LSH(query_vector)
        query_hash_key_list = self.fetch_hash_keys(query_bucket_list)
        query_hash_key_list = list(set(query_hash_key_list))
        print("Hash Key List {0}".format(query_hash_key_list))
        selected_image_set = set()
        nearest_neighbour_list = set()
        total_images_considered = []

        for j in range(0, self.num_hashs):
            for bucket in query_hash_key_list:
                print("Getting bucket - {0}".format(bucket.rsplit("-", j)[0]))
                images_in_current_bucket = self.index_structure.get(bucket.rsplit("-", j)[0], [''])
                images_in_current_bucket = set(images_in_current_bucket)
                images_in_current_bucket.discard('')
                selected_image_set.update(images_in_current_bucket)
                total_images_considered.extend(list(images_in_current_bucket))
                feature_vectors = []
                for img in selected_image_set:
                    feature_vectors.append(list(mongo_client.mwdb_project.image_features.find({'imageName': img}))[0]["HOG_reduced"])

                for img, fv in zip(selected_image_set, feature_vectors):
                    eucledian_distance = distance.euclidean(fv, query_vector)
                    if (eucledian_distance != 0):
                        nearest_neighbour_list.add((img, eucledian_distance))

        #    if len(nearest_neighbour_list) >= no_of_images:
        #        break

        nearest_neighbour_list = sorted(nearest_neighbour_list, key=lambda x: x[1])

        return nearest_neighbour_list[:no_of_images], len(nearest_neighbour_list),len(total_images_considered)
