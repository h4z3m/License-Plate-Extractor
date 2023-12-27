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


class Segmentation():

    @classmethod
    def segment_image(cls, image):
        # Read and resize the image
        # Convert RGBA to grayscale
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        gray_image = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2GRAY)
        # gray_image = cv2.resize(gray_image, (20, 20))
        # Create empty array to store all ROI images
        roi_images = []

        # Apply thresholding
        threshold = filters.threshold_otsu(gray_image)
        thresholded_image = np.zeros_like(gray_image)
        thresholded_image[gray_image > threshold] = 1

        # Find contours
        contours, hierarchy = cv2.findContours(
            thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours from left to right
        sorted_contours = sorted(
            contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        # Create empty array to store all ROI images
        roi_images = []

        # Iterate through contours and draw bounding boxes
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Create ROI images
            roi = thresholded_image[y:y+h, x:x+w]
            # if cls.check_valid_contour(cnt, image.shape):
            roi_images.append(roi)

        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 3, 1)
        # plt.imshow(image)
        # plt.title('Original Image')

        # plt.subplot(1, 3, 2)
        # plt.imshow(thresholded_image, cmap='gray')
        # plt.title('Resized Image')

        # # plt.subplot(1, 3, 3)
        # # plt.imshow(thresholded_image, cmap='gray')
        # # plt.title('Thresholded Image')
        # plt.show()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(roi_images)+1, 1)
        plt.imshow(image, cmap='gray')

        # Iterate over the ROI images and plot them
        for i, image in enumerate(roi_images):
            plt.subplot(1, len(roi_images)+1, i+2)
            plt.imshow(image, cmap='gray')
            plt.title(f'ROI Image {i+1}')

        plt.show()

        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, sorted_contours, -1,
                         (0, 255, 0), 2)  # Green color, thickness=2

        print(len(sorted_contours))
        # Display the image with contours
        cv2.imshow('Image with Contours', image_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return roi_images

    @classmethod
    def check_valid_contour(cls, contour, image_shape):
        # Check if contour is a closed contour

        # Check if the contour is big enough
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        image_area = image_shape[0] * image_shape[1]

        if contour_area < 0.01 * image_area:
            return False

        return True


if __name__ == "__main__":
    image = io.imread(
        "C:/Users/10/OneDrive/Desktop/Data Set/passed/0588[00].jpg")
    Segmentation.segment_image(image)
