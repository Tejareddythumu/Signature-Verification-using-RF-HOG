import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import joblib

# Function to load and preprocess images
def load_images(root_dir):
    allowed_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    image_pairs = []
    labels = []

    unique_signature_folders = sorted(os.listdir(root_dir), key=lambda x: int(x))

    for unique_signature_folder in unique_signature_folders:
        unique_folder_path = os.path.join(root_dir, unique_signature_folder)

        if os.path.isdir(unique_folder_path):
            genuine_path = os.path.join(unique_folder_path, 'full_org')
            forged_path = os.path.join(unique_folder_path, 'full_forg')

            if not os.path.exists(genuine_path) or not os.path.exists(forged_path):
                continue

            genuine_files = sorted([f for f in os.listdir(genuine_path) if f.lower().endswith(allowed_extensions)])
            forged_files = sorted([f for f in os.listdir(forged_path) if f.lower().endswith(allowed_extensions)])

            for gen in genuine_files:
                gen_path = os.path.join(genuine_path, gen)
                for forg in forged_files:
                    forg_path = os.path.join(forged_path, forg)
                    image_pairs.append((gen_path, forg_path))
                    labels.append(0)  # Forged pair
                image_pairs.append((gen_path, gen_path))
                labels.append(1)  # Genuine pair

    return image_pairs, labels

# Function to extract features using HOG
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize((128, 128))
        image = np.array(image)
        features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        return features
    except Exception as e:
        return None

# Main script
root_dir = "D:\\PyCharm 2024.3.5\\signature_verification\\test"  # Adjust this path as needed
image_pairs, labels = load_images(root_dir)

features = []
for img1_path, img2_path in image_pairs:
    features1 = extract_features(img1_path)
    features2 = extract_features(img2_path)
    if features1 is not None and features2 is not None:
        combined_features = np.concatenate((features1, features2))
        features.append(combined_features)

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_classifier = grid_search.best_estimator_

# Save the model
joblib.dump(best_rf_classifier, "signature_verification_model_test.pkl")
# Save the scaler
joblib.dump(scaler, "scaler_test.pkl")