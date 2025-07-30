import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the pre-trained ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add a GlobalMaxPooling2D layer to the model
feature_extractor_model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, feature_model):
    """Extracts features from an image using the specified feature extraction model."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = feature_model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Update to specify the image folder path in the same directory as app.py
image_folder = os.path.join(os.getcwd(), 'Images')

# Collect all image file paths from nested directories
filenames = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        # Adjust the file extension check if needed (e.g., for .tif, .jpg, .png, etc.)
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff')):
            filenames.append(os.path.join(root, file))

# Extract features for each image and save them using pickle
feature_list = []
for file in tqdm(filenames):
    features = extract_features(file, feature_extractor_model)
    if features is not None:
        feature_list.append(features)

# Save the feature list and filenames to files using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))