import os
import numpy as np
import cv2
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Set directories
train_dir = './train/'  # Path to the train folder
test_dir = './test1/'   # Path to the test1 folder
output_csv = 'image_classifications.csv'  # Output CSV file

# Helper function to load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # Read the image
            img = cv2.resize(img, (64, 64))  # Resize to a fixed size (e.g., 64x64)
            img = img.flatten()  # Flatten the image to a 1D array
            images.append(img)
            labels.append(label)
    return images, labels

# Load training data
cats, cat_labels = load_images_from_folder(os.path.join(train_dir, 'cat'), 0)
dogs, dog_labels = load_images_from_folder(os.path.join(train_dir, 'dog'), 1)

# Combine images and labels
X = np.array(cats + dogs)
y = np.array(cat_labels + dog_labels)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Optional: Use PCA for dimensionality reduction (improves SVM performance)
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Create and train the SVM classifier
svm = SVC(kernel='linear')  # Linear kernel for binary classification
svm.fit(X_train_pca, y_train)

# Function to predict and classify an image from test1 folder
def predict_image(image_number, model, scaler, pca):
    img_name = f"{image_number}.jpg"
    img_path = os.path.join(test_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Error: The image {img_name} was not found in the {test_dir} folder.")
        return None
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Resize to the same size as training images
    img = img.flatten().reshape(1, -1)  # Flatten the image into a 1D array
    img_scaled = scaler.transform(img)
    img_pca = pca.transform(img_scaled)
    prediction = model.predict(img_pca)
    
    return 'cat' if prediction == 0 else 'dog'

# Function to classify all images in test1 folder and save results to CSV
def classify_all_images(model, scaler, pca, test_dir, output_csv):
    results = []

    for filename in os.listdir(test_dir):
        if filename.endswith('.jpg'):
            image_number = int(filename.split('.')[0])
            category = predict_image(image_number, model, scaler, pca)
            
            if category:
                results.append([image_number, category])
    
    # Save the results to a CSV file
    df = pd.DataFrame(results, columns=['image', 'category'])
    df.to_csv(output_csv, index=False)
    print(f"All images classified and results saved to {output_csv}")

# First, classify all images in the test1 folder and save results to CSV
classify_all_images(svm, scaler, pca, test_dir, output_csv)

# Now, repeatedly ask the user for an image number until they type 'stop'
while True:
    user_input = input("\nEnter the image number (e.g., 2 for 2.jpg) or type 'stop' to exit: ")
    
    if user_input.lower() == 'stop':
        print("Exiting the image classification.")
        break

    try:
        image_number = int(user_input)
        result = predict_image(image_number, svm, scaler, pca)

        if result:
            print(f"The image {image_number}.jpg is classified as: {result}")
    except ValueError:
        print("Invalid input. Please enter a valid image number or 'stop' to exit.")
