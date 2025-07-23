import streamlit as st
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,
preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

# Function to convert .tif to .jpg if necessary
def load_and_convert_image(img_path):
    img = Image.open(img_path)
    if img.format != 'JPEG':
        # Convert the image to JPEG
        img = img.convert('RGB')
        img_path = img_path.replace('.tif', '.jpg')
        img.save(img_path, 'JPEG')
    return img_path

# Function to load image as bytes for Streamlit
def load_image_bytes(img_path):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    return img_bytes

# Load precomputed features and file list
features = pickle.load(open("embeddings.pkl", "rb"))
img_files_list = pickle.load(open("filenames.pkl", "rb"))

# Normalize the file paths
img_files_list = [os.path.normpath(filename).replace("\\", "/") for filename in img_files_list]

# Load ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False