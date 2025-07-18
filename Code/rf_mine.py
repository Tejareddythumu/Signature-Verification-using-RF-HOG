import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess images with limited pairs from all unique signature subfolders
def load_images(root_dir):
    allowed_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    image_pairs = []
    labels = []

    # Filter out non-numeric folder names
    unique_signature_folders = [
        folder for folder in os.listdir(root_dir) if folder.isdigit()
    ]
    unique_signature_folders.sort(key=int)

    print(f"Unique signature folders found: {unique_signature_folders}")

    for unique_signature_folder in unique_signature_folders:
        unique_folder_path = os.path.join(root_dir, unique_signature_folder)
        print(f"Checking folder: {unique_folder_path}")

        if os.path.isdir(unique_folder_path):
            genuine_path = os.path.join(unique_folder_path, 'full_org')
            forged_path = os.path.join(unique_folder_path, 'full_forg')

            if not os.path.exists(genuine_path):
                print(f"Genuine path does not exist: {genuine_path}")
                continue
            if not os.path.exists(forged_path):
                print(f"Forged path does not exist: {forged_path}")
                continue

            genuine_files = sorted([f for f in os.listdir(genuine_path) if f.lower().endswith(allowed_extensions)])
            forged_files = sorted([f for f in os.listdir(forged_path) if f.lower().endswith(allowed_extensions)])

            print(f"Found {len(genuine_files)} genuine files in {genuine_path}.")
            print(f"Found {len(forged_files)} forged files in {forged_path}.")

            for gen in genuine_files:
                gen_path = os.path.join(genuine_path, gen)
                for forg in forged_files:
                    forg_path = os.path.join(forged_path, forg)
                    image_pairs.append((gen_path, forg_path))
                    labels.append(0)
                image_pairs.append((gen_path, gen_path))
                labels.append(1)

    print(f"Total pairs generated: {len(image_pairs)}")
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
        print(f"Error loading image {image_path}: {e}")
        return None

# Save all output to a text file
with open("results_mine.txt", "w") as f:
    with redirect_stdout(f):
        root_dir = "D:\\PyCharm 2024.3.5\\signature_verification\\my_sign"
        image_pairs, labels = load_images(root_dir)

        if not image_pairs:
            print("No image pairs found. Please check the directory structure and image files.")
        else:
            features = []
            for img1_path, img2_path in image_pairs:
                features1 = extract_features(img1_path)
                features2 = extract_features(img2_path)
                if features1 is not None and features2 is not None:
                    combined_features = np.concatenate((features1, features2))
                    features.append(combined_features)

            features = np.array(features)
            labels = np.array(labels)

            # Check class distribution
            unique, counts = np.unique(labels, return_counts=True)
            print(f"Label distribution: {dict(zip(unique, counts))}")

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
            y_pred = best_rf_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

            cv_scores = cross_val_score(best_rf_classifier, features, labels, cv=5)
            print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

            conf_matrix = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(conf_matrix)

            print("Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

# Save the confusion matrix image
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Forged', 'Genuine'], yticklabels=['Forged', 'Genuine'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_custom.png")
plt.close()
