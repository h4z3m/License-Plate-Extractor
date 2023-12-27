import cv2
from matplotlib import pyplot as plt
import numpy as np
import skimage.io as io
from skimage import filters

from skimage.segmentation import clear_border

from utils.utils import debug_plot_image


def plot_image(img, title=""):
    """
    Plots an image using matplotlib.

    Parameters:
        img (numpy.ndarray): The image to be plotted.
        title (str, optional): The title of the plot. Defaults to an empty string.

    Returns:
        None
    """
    plt.figure(figsize=[7, 7])
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


class Segmentation:
    @classmethod
    def segment_image(cls, original, image):
        # Read and resize the image
        # Convert RGBA to grayscale
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        gray_image = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2GRAY)
        # Sharpen the image
        hpf = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        gray_image = cv2.filter2D(gray_image, -1, hpf)
        # gray_image = cv2.resize(gray_image, (20, 20))
        # Create empty array to store all ROI images
        roi_images = []

        # Remove borders 5px left and right
        gray_image = gray_image[:, 2:-2]
        # Apply thresholding
        threshold_otsu = filters.threshold_otsu(gray_image)

        thresholded_image = np.zeros_like(gray_image)

        otsu_thresholded = np.zeros_like(gray_image)

        otsu_thresholded[gray_image > threshold_otsu] = 1
        otsu_thresholded = 1 - otsu_thresholded
        ####### yen ##########
        yen_thresh = filters.threshold_yen(gray_image)
        gradient = np.zeros_like(gray_image)
        gradient[gray_image < yen_thresh] = 1

        gradient = cv2.erode(
            gradient,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )

        gradient = cv2.dilate(
            gradient,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4)),
        )

        middle_part = otsu_thresholded.copy()
        middle_part[:, : gray_image.shape[1] // 2 - 3] = 0
        middle_part[:, gray_image.shape[1] // 2 + 3 :] = 0
        # plot_image(middle_part)
        lines = cv2.erode(
            middle_part,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, otsu_thresholded.shape[1])),
        )

        # otsu_thresholded = otsu_thresholded - middle_part

        # erode grad
        # Remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        otsu_thresholded = cv2.erode(otsu_thresholded, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 4))
        otsu_thresholded = cv2.dilate(otsu_thresholded, kernel, iterations=2)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        # otsu_thresholded = cv2.erode(otsu_thresholded, kernel)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        # # otsu_thresholded = cv2.dilate(otsu_thresholded, kernel, iterations=1)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        # otsu_thresholded = cv2.erode(otsu_thresholded, kernel, iterations=1)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # otsu_thresholded = cv2.dilate(otsu_thresholded, kernel, iterations=1)

        # otsu_thresholded = cv2.dilate(otsu_thresholded, kernel)

        # otsu_thresholded = cv2.erode(otsu_thresholded, kernel)
        # otsu_thresholded = cv2.erode(otsu_thresholded, kernel)

        thresholded_image = otsu_thresholded

        debug_plot_image(gradient, "gradient")
        # plt.figure()
        # plt.imshow(gradient, cmap="gray")

        # plt.figure()
        # plt.imshow(otsu_thresholded, cmap="gray")

        # Find contours
        contours, hierarchy = cv2.findContours(
            gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # show contours

        contours_image = cv2.drawContours(original.copy(), contours, -1, (0, 255, 0), 1)
        # plot_image(lines)
        # plot_image(contours_image)
        # Sort contours from left to right
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        # Create empty array to store all ROI images
        roi_images = []

        # Iterate through contours and draw bounding boxes
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Create ROI images
            roi = gradient[y : y + h, x : x + w]
            if cls.check_valid_contour(cnt, image.shape):
                roi_images.append(roi)

        return roi_images

    @classmethod
    def check_valid_contour(cls, contour, image_shape):
        # Check if contour is a closed contour

        # Check if the contour is big enough
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        image_area = image_shape[0] * image_shape[1]

        if w * h < 0.02 * image_area or w * h > 0.8 * image_area:
            return False

        return True


if __name__ == "__main__":
    image = io.imread("C:/Users/10/OneDrive/Desktop/Data Set/passed/0588[00].jpg")
    Segmentation.segment_image(image)
