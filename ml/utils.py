import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json


# save list of sorted images to json for frontend
def save_to_json(sorted_arr, output_path = '../public/data/sortedImages.json'):
    with open(output_path, 'w') as f:
        json.dump(sorted_arr, f, indent=2)


# check for multiple objects and if is white blob
def is_valid_mask(image_path, max_fill_ratio=0.65):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   
   # binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    white_ratio = np.sum(binary == 255) / binary.size
    
    # get contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter out images with more than 1 contour (object)
    if len(contours) != 1:
        return False

    # filter out white blob images
    if white_ratio > max_fill_ratio:
        return False
    return True


# get hu moments
def extract_hu_moments(image_path, output_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # call this after validating with is_valid_mask so will always have 1 contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]  

    # crop image to bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = binary[y:y+h, x:x+w]

    # resize and create new blank canvas for object
    scale = min(output_size[0] / h, output_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(output_size, dtype=np.uint8)
    x_offset = (output_size[1] - new_w) // 2
    y_offset = (output_size[0] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # get Hu Moments
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    # need to return log version for scaling/ normalizing numbers
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)



# attempt to sort image shapes by on hu moments
def sort_images(folder_path):
    all_files = sorted(os.listdir(folder_path))
    image_paths = [os.path.join(folder_path, f) for f in all_files if f.lower().endswith((".png"))]
    
    valid_paths = []
    features = []

    for path in image_paths:
        if is_valid_mask(path):
            hu = extract_hu_moments(path)
            if hu is not None:
                valid_paths.append(path)
                features.append(hu)

    # print(features[0])

    pca = PCA(n_components=7)
    scores = pca.fit_transform(np.array(features))[:, 0]    
    sorted_indices = np.argsort(scores)
    # sorted_paths = [valid_paths[i] for i in sorted_indices]
    sorted_filenames = [os.path.basename(valid_paths[i]) for i in sorted_indices]

    return sorted_filenames