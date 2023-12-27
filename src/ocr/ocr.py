import cv2
import os
import skimage.io as io
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from preprocessing import gray_image, HistogramEqualization
from scipy.io import loadmat
from skimage import filters
import numpy as np
import pickle


class OCR():
    training_dataset = []
    training_dataset_labels = []

    # Classifiers
    classifier = None

    def load_dataset(self, training_directory):
        self.training_dataset_labels = []
        self.training_dataset = []

        for filename in os.listdir(training_directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Assuming the images are in JPEG or PNG format
                image_path = os.path.join(
                    training_directory, filename).encode("utf8").decode()
                # Read the image
                try:
                    image = io.imread(image_path)
                except Exception as e:
                    # print(f"Error loading image {image_path}: {e}")
                    continue

                if image is None:
                    # print(f"Error loading image {image_path}")
                    continue

                resized_img = cv2.resize(image, (16, 32))
                grayscale_thresholded_image = cv2.cvtColor(
                    (resized_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                threshold = filters.threshold_otsu(grayscale_thresholded_image)
                thresholded_image = np.zeros(resized_img.shape)
                thresholded_image[resized_img > threshold] = 1

                # Apply HOG on the grayscale image
                feature_vector = hog(grayscale_thresholded_image, pixels_per_cell=(
                    2, 4), cells_per_block=(2, 4))

                # Extract label from the filename or any other source
                label = self.extract_label_from_filename(filename)

                self.training_dataset_labels.append(label)
                self.training_dataset.append(feature_vector)

    def extract_label_from_filename(self, filename):
        part = filename.split('-')[1]
        return part

    def train(self, mode: str):
        if mode == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=5000)
        elif mode == "svm":
            self.classifier = SVC(kernel='rbf')
        elif mode == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100, criterion='entropy', random_state=0)

        self.classifier.fit(self.training_dataset,
                            self.training_dataset_labels)

    def save_trained_model(self, pickle_file_path="trained_model.pk1"):
        # delete file contents in trained_model.pk1
        open(pickle_file_path, 'w').close()

        # save the model to disk
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(self.classifier, file)

    def load_trained_model(self, pickle_file_path="trained_model.pk1"):
        # load the model from disk
        with open(pickle_file_path, 'rb') as file:
            self.classifier = pickle.load(file)

    def predict(self, img_to_predict):
        resized_img = cv2.resize(img_to_predict, (16, 32))
        grayscale_thresholded_image = cv2.cvtColor(
            (resized_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        feature_vector = hog(grayscale_thresholded_image, pixels_per_cell=(
            2, 4), cells_per_block=(2, 4))
        return self.classifier.predict([feature_vector])


if __name__ == "__main__":
    classifier = OCR()
    classifier.load_dataset(
        "C:\\Users\\10\\OneDrive\\Desktop\\Data Set\\Characters")

    # # Train the classifier (choose 'knn', 'svm', or 'rf' for the mode)
    classifier.train(mode="svm")

    test_image = io.imread(
        "C:/Users/10/OneDrive/Desktop/Data Set/test/Screenshot 2023-12-26 074037.jpg.png")

    # # save the trained model
    classifier.save_trained_model()

    # load the trained model
    # classifier.load_trained_model()

    # predict the test image
    predicted_label = classifier.predict(test_image)

    file_path = "output.txt"

    with open(file_path, 'w', encoding='utf-8') as file:
        # file.write(predicted_digit + '\n\n')
        np.savetxt(file, predicted_label, fmt='%s', delimiter='\t')
