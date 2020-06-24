from skimage import transform, feature, io
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import skew
import cv2
from project_utils import *

def calculate_first_moment(channel):
    return np.mean(channel)

def calculate_second_moment(channel):
    variance = np.var(channel)
    return np.sqrt(variance)

def calculate_third_moment(channel):
    # Flatten the array, and calculate the skew
    return skew(np.ndarray.flatten(channel), 0)

# Method that extracts the color moments, reads image from disk, converts to YUV,
# splits it into 100 x 100 windows then extracts color moments for each channel for each window.
def extract_color_moments(path_to_image):
    mongo_client = connect_to_db()
    image_id = os.path.basename(path_to_image)
    img = cv2.imread(path_to_image)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(img_yuv)
    color_moments = []
    for channel in [y, u, v]:
        slices = slice_channel(channel)
        for channel_slice in slices:
            color_moments.append(calculate_first_moment(channel_slice))
            color_moments.append(calculate_second_moment(channel_slice))
            color_moments.append(calculate_third_moment(channel_slice))

    image_features = mongo_client.mwdb_project.image_features
    image = image_features.update_one({'imageName': image_id}, { '$set' : {'image_path' : path_to_image, 'CM' : color_moments}  }, upsert=True)
    mongo_client.close()
    return color_moments

# Method to extract Lbp histograms, read image from disk, conver to grayscale,
# split to 100 x 100 windows, compute LBP and then the histogram of the LBP image.
def extract_lbp_features(path_to_image):
    image_id = os.path.basename(path_to_image)
    img = cv2.imread(path_to_image)
    img_greyscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    slices = slice_channel(img_greyscale)
    lbp_features = []
    for channel_slice in slices:
        # LBP Is calculated with radius = 1 and points  =  3, method = "default"
        lbp = local_binary_pattern(img_greyscale, 1, 3, 'default')
        hist, edges = np.histogram(lbp, bins = 10)
        hist = hist.astype("float")
        hist = hist/(hist.sum() + 1e-7)
        lbp_features.extend(hist.tolist())
    mongo_client = connect_to_db()
    image = mongo_client.mwdb_project.image_features.update_one({'imageName': image_id}, {'$set' : { 'image_path' : path_to_image, 'LBP' : lbp_features }}, upsert=True)
    mongo_client.close()
    return lbp_features

def extract_sift_features(path_to_image):
    image_id = os.path.basename(path_to_image)
    img = cv2.imread(path_to_image)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, feature_vector = sift.detectAndCompute(img, None)
    points = np.array([[kp[idx].pt[0], kp[idx].pt[1], kp[idx].size, kp[idx].angle, kp[idx].response] for idx in range(0, len(kp))])
    feature_vector = np.concatenate((feature_vector, points), axis = 1)

    # here 132 means, we're keeping x, y, scale, orientation and excluding the response, we don't want it.
    feature_vector = np.flip(feature_vector[feature_vector[:, -1].argsort()][:, :132], axis=0)

    mongo_client = connect_to_db()
    image = mongo_client.mwdb_project.image_features.update_one({'imageName': image_id}, {'$set' : { 'image_path' : path_to_image, 'SIFT' :feature_vector.tolist()}}, upsert=True)
    mongo_client.close()

    return feature_vector

def extract_hog_features(path_to_image):
    image_id = os.path.basename(path_to_image)
    img = cv2.imread(path_to_image)
    # downsampling the image by factor of 1 per 10.
    downscaled_image = transform.downscale_local_mean(img, (10,10,1))
    feature_vector, hog_image = feature.hog(downscaled_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)

    mongo_client = connect_to_db()
    image = mongo_client.mwdb_project.image_features.update_one({'imageName': image_id}, {'$set' : { 'image_path' : path_to_image, 'HOG' : feature_vector.tolist()}}, upsert=True)
    mongo_client.close()

    return feature_vector
