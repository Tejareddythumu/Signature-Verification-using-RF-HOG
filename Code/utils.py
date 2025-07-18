# utils.py
'''import numpy as np
from PIL import Image
from skimage.feature import hog

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((128, 128))
        image = np.array(image)
        features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        return features
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
'''
import numpy as np
from PIL import Image
from skimage.feature import hog

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((128, 128))
        image = np.array(image)
        features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        return features
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None