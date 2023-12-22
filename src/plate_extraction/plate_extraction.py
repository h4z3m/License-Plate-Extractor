import json
from debug_logger.debug_logger import DebugLogger
import cv2
import imutils
import numpy as np
import utils.utils as utils
import logging
from typing import Tuple, List

logger = DebugLogger(name="logger")

# Load configuration from JSON file
with open("./config/plate_extraction_config.json") as config_file:
    logger.debug("Loading plate extraction config")
    module_config = json.load(config_file)

minAR = module_config.get("min_aspect_ratio", 1.5)
maxAR = module_config.get("max_aspect_ratio", 3.5)
perfectAR = module_config.get("perfect_aspect_ratio", 3.5)
minArea = module_config.get("min_area", 600)
maxArea = module_config.get("max_area", 10_000)
minHeight = module_config.get("min_height", 10)
maxHeight = module_config.get("max_height", 55)
minWidth = module_config.get("min_width", 10)
maxWidth = module_config.get("max_width", 100)
colorRegionToPlateHeightRatio = module_config.get(
    "color_region_to_plate_height_ratio", 0.65
)

logger.debug(
    f""" Loaded plate extraction config
    minAr: {minAR}
    maxAr: {maxAR}
    perfectAr: {perfectAR}
    minArea: {minArea}
    maxArea: {maxArea}
    minHeight: {minHeight}
    maxHeight: {maxHeight}
    minWidth: {minWidth}
    maxWidth: {maxWidth}
    colorRegionToPlateHeightRatio: {colorRegionToPlateHeightRatio}
    """
)

"""""" """""" " Contour detection " """""" """"""


def get_contours(image, top=1, sort_fn=cv2.contourArea):
    """
    Given an image, this function finds and returns the specified number of contours.

    Parameters:
        image (numpy.ndarray): The input image.
        top (int, optional): The number of contours to return. Defaults to 1.
        sort_fn (function, optional): The function used to sort the contours.
            Defaults to cv2.contourArea.

    Returns:
        list: A list of contours.
    """
    contours = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=sort_fn, reverse=True)[:top]
    logger.debug(f"Found {len(contours)} contours")
    return contours


"""""" """""" " Candidate contour extraction " """""" """"""


def area_AR_get_candidate_contours(contours):
    """
    Generate the candidate contours based on the given parameters.

    Parameters:
        contours (list): List of contours to process.

    Returns:
        list: List of candidate contours.
    """
    main_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if (
            minAR <= ar <= maxAR
            and minHeight <= h <= maxHeight
            and minArea <= w * h <= maxArea
        ):
            main_contours.append(c)
    logger.debug(f"Found {len(main_contours)} candidate contours")
    return main_contours


def area_binary_thresh_get_candidate_contours(
    contours: list, image: np.ndarray, gray: np.ndarray
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Calculate the set of candidate contours from a list of contours based on binary thresholding.

    Parameters:
        contours (list): A list of contours.
        image (np.ndarray): The original image.
        gray (np.ndarray): The grayscale image.

    Returns:
        List[Tuple(np.ndarray, float, float)]: The list of candidate contours, each represented as a tuple containing the contour image, the area ratio, and the white pixel percentage.
    """
    images_set = []
    _, binary_image = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    # Invert the image so that letters are black and background is white
    result_image = cv2.bitwise_or(gray, binary_image)
    for c in contours:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        area_ratio = area / (w * h)
        ar = w / float(h)
        if (
            (minAR <= ar <= maxAR)
            and (minHeight <= h)
            and (minArea <= w * h <= maxArea)
            and (minWidth < w)
        ):
            licensePlate = result_image[y : y + h, x : x + w]
            _, binary_mask = cv2.threshold(licensePlate, 165, 255, cv2.THRESH_BINARY)
            white_pixel_percentage = (
                np.count_nonzero(binary_mask == 255) / binary_mask.size
            )
            images_set.append(
                (
                    image[y : y + h, x : x + w],
                    area_ratio,
                    white_pixel_percentage,
                    (x, y, w, h),
                )
            )
    return images_set


"""""" """""" " License plate detection " """""" """"""


def perfectAR_method_plate_detection(gray, contours: list):
    """
    Finds the perfect AR plate detection based on the given gray image and contours.

    Args:
        gray (numpy.ndarray): The gray image.
        contours (list): A list of contours.

    Returns:
        numpy.ndarray: The license plate image.
    """
    min_diff = float("inf")
    x_final = 0
    y_final = 0
    w_final = 0
    h_final = 0
    # Iterate over the contours
    for c in contours:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        """# candidate = gray[y : y + h, x : x + w]
        # hist = cv2.calcHist([candidate], [0], None, [256], [0, 256])
        # sum = np.sum(hist[160:256])
        # if sum < 0.6 * w * h:
        #     continue"""

        # Calculate the aspect ratio of the bounding rectangle
        ar = w / float(h)
        # Calculate the difference between the aspect ratio of the bounding rectangle
        # and the aspect ratio of a perfect rectangle
        diff = abs(ar - perfectAR)
        # If this difference is smaller than the smallest difference we've seen so far,
        # update the most rectangular contour and the smallest difference
        if diff < min_diff:
            min_diff = diff
            x_final = x
            y_final = y
            w_final = w
            h_final = h
    licensePlate = gray[y_final : y_final + h_final, x_final : x_final + w_final]
    return licensePlate


def hog_method_plate_detection(images_set, reference_features):
    """
    Compute the plate detection using the Histogram of Oriented Gradients (HOG) method.

    Parameters:
    - images_set (list): A list of tuples containing the images, their bounding boxes, and their labels.
    - reference_features (ndarray): An array of HOG features of the reference image.

    Returns:
    - nearest_images (list): A list of tuples containing the nearest images to the reference image, along with their bounding boxes and labels.
    """
    hog = cv2.HOGDescriptor()

    distances = []
    for img, _, _, _ in images_set:
        # Convert image to grayscale
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, (128, 128))

        # Compute HOG features for the current image
        image_features = hog.compute(image_gray)

        # Calculate Euclidean distance
        distance = np.linalg.norm(reference_features - image_features)
        distances.append(distance)
        logger.info(f"Distance: {distance}")

    nearest_indices = np.argsort(distances)[:2]
    nearest_images = [images_set[i] for i in nearest_indices]
    logger.info(f"Found nearest images: {len(nearest_images)}")
    return nearest_images


def extract_color_region(original_image, x, y, w, h):
    """
    Extracts the color region from the original image based on the given coordinates and dimensions.

    Args:
        original_image (numpy.ndarray): The original image from which to extract the colored region.
        x (int): The x-coordinate of the top-left corner of the region.
        y (int): The y-coordinate of the top-left corner of the region.
        w (int): The width of the region.
        h (int): The height of the region.

    Returns:
        Tuple[int, int, int]: The indices of the maximum value in the histogram calculated from the colored region.
    """
    colored_region = original_image[
        y - round(colorRegionToPlateHeightRatio * h) : y, x : x + w
    ]
    # Calculate the histogram
    hist = cv2.calcHist(
        [colored_region], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )

    # Find the indices of the maximum value in the histogram
    max_indices = np.unravel_index(hist.argmax(), hist.shape)
    logger.debug(f"Max indices: {max_indices}")
    if logger.getEffectiveLevel() == logging.DEBUG:
        block = np.ones((10, 10, 3), np.uint8)
        block[...] = max_indices
        utils.plot_image(block)
    return max_indices
