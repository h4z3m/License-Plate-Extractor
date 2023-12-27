import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage import filters
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class OCR:
    training_dataset = []
    training_dataset_labels = []

    # Classifiers
    classifier = None

    def load_dataset(self, training_directory):
        """
        Load the dataset from the given training directory.

        Parameters:
            training_directory (str): The path to the directory containing the training images.

        Returns:
            None
        """
        self.training_dataset_labels = []
        self.training_dataset = []

        for filename in os.listdir(training_directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Assuming the images are in JPEG or PNG format
                image_path = (
                    os.path.join(training_directory, filename).encode("utf8").decode()
                )
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
                    (resized_img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
                )
                threshold = filters.threshold_otsu(grayscale_thresholded_image)
                thresholded_image = np.zeros(resized_img.shape)
                thresholded_image[resized_img > threshold] = 1

                # Apply HOG on the grayscale image
                feature_vector = hog(
                    grayscale_thresholded_image,
                    pixels_per_cell=(2, 4),
                    cells_per_block=(2, 4),
                )

                # Extract label from the filename or any other source
                label = self.extract_label_from_filename(filename)

                self.training_dataset_labels.append(label)
                self.training_dataset.append(feature_vector)

    def extract_label_from_filename(self, filename):
        """
        Extracts the label from a given filename.

        Parameters:
        - `filename` (str): The name of the file from which to extract the label.

        Returns:
        - `str`: The extracted label.
        """
        part = filename.split("-")[1]
        return part

    def train(self, mode: str):
        """
        Train the classifier using the specified mode.

        Parameters:
            mode (str): The mode for training the classifier. Valid options are "knn", "svm", or "rf".

        Returns:
            None
        """
        if mode == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=5000)
        elif mode == "svm":
            self.classifier = SVC(kernel="rbf")
        elif mode == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100, criterion="entropy", random_state=0
            )

        self.classifier.fit(self.training_dataset, self.training_dataset_labels)

    def save_trained_model(self, pickle_file_path="trained_model.pk1"):
        """
        Saves the trained model to a pickle file.

        Parameters:
            pickle_file_path (str): The path to the pickle file. Defaults to "trained_model.pk1".

        Returns:
            None
        """
        # delete file contents in trained_model.pk1
        open(pickle_file_path, "w").close()

        # save the model to disk
        with open(pickle_file_path, "wb") as file:
            pickle.dump(self.classifier, file)

    def load_trained_model(self, pickle_file_path="trained_model.pk1"):
        """
        Load a trained model from a pickle file.

        Parameters:
            pickle_file_path (str): The path to the pickle file containing the trained model.
                                   Default is "trained_model.pk1".

        Returns:
            None
        """
        # load the model from disk
        with open(pickle_file_path, "rb") as file:
            self.classifier = pickle.load(file)

    def predict(self, img_to_predict):
        """
        Generates a prediction for the given image.

        Parameters:
            img_to_predict (numpy.ndarray): The image to be predicted.

        Returns:
            numpy.ndarray: The predicted class label for the image.
        """
        resized_img = cv2.resize(img_to_predict, (16, 32))

        feature_vector = hog(
            resized_img, pixels_per_cell=(2, 4), cells_per_block=(2, 4)
        )
        return self.classifier.predict([feature_vector])


if __name__ == "__main__":
    classifier = OCR()
    classifier.load_dataset("../../data/Characters")

    # # Train the classifier (choose 'knn', 'svm', or 'rf' for the mode)
    classifier.train(mode="rf")

    test_image = io.imread("../../data/Testing_output/passed_800_900/0896[00].jpg")

    # # save the trained model
    classifier.save_trained_model("trained_model_rf.pk1")

    # load the trained model
    # classifier.load_trained_model()

    # predict the test image
    predicted_label = classifier.predict(test_image)

    file_path = "output.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        # file.write(predicted_digit + '\n\n')
        np.savetxt(file, predicted_label, fmt="%s", delimiter="\t")
