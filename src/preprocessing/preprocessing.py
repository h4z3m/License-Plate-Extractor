import numpy as np
import cv2
from scipy.signal import convolve2d


def sobel(
    img: np.ndarray,
    threshold: int = 0,
    dir="both",
    hx=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    hy=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
):
    """
    Calculates the Sobel filter of the input image.

    Parameters:
        img (np.ndarray): The input image.
        threshold (int, optional): The threshold value. Defaults to 0.
        dir (str, optional): The direction of the filter. Can be "both", "horizontal", or "vertical". Defaults to "both".
        hx (np.ndarray, optional): The horizontal Sobel kernel. Defaults to np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).
        hy (np.ndarray, optional): The vertical Sobel kernel. Defaults to np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).

    Returns:
        np.ndarray: The filtered image.
    """
    final_image: np.ndarray
    if dir == "both":
        horizontal = convolve2d(img, hy)
        vertical = convolve2d(img, hx)
        final_image = np.hypot(horizontal, vertical)
    elif dir == "horizontal":
        final_image = convolve2d(img, hy)
    elif dir == "vertical":
        final_image = convolve2d(img, hx)
    else:
        raise ValueError(f"dir must be either 'both', 'horizontal' or 'vertical'")
    removed = np.where(np.abs(final_image) * 255 < threshold)
    final_image[removed] = 0
    return final_image


def extract_hog_features(
    img,
    target_img_size=(32, 32),
    win_size=(32, 32),
    cell_size=(4, 4),
    block_size_in_cells=(2, 2),
):
    """
    Extracts Histogram of Oriented Gradient (HOG) features from an image.

    Args:
        img (numpy.ndarray): The input image.
        target_img_size (tuple): The size to which the image should be resized
            before feature extraction. Defaults to (32, 32).
        win_size (tuple): The size of the detection window. Defaults to (32, 32).
        cell_size (tuple): The size of each cell in the detection window.
            Defaults to (4, 4).
        block_size_in_cells (tuple): The size of each block in terms of the number
            of cells. Defaults to (2, 2).

    Returns:
        numpy.ndarray: The extracted HOG features as a flattened array.
    """
    img = cv2.resize(img, target_img_size)

    block_size = (
        block_size_in_cells[1] * cell_size[1],
        block_size_in_cells[0] * cell_size[0],
    )
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def get_lbp_hist(grayscale_img):
    """
    Calculate the Local Binary Pattern (LBP) histogram of a grayscale image.

    Args:
        grayscale_img (ndarray): The input grayscale image.

    Returns:
        ndarray: The LBP histogram, represented as a 256-element array.

    """

    img = grayscale_img
    lbp = np.zeros(256)
    for i in range(
        1,
        img.shape[0] - 1,
    ):
        for j in range(
            1,
            img.shape[1] - 1,
        ):
            patch = img[i - 1 : i + 2, j - 1 : j + 2]
            value = (
                ((patch[0, 0] > patch[1, 1]) << 7)  # top left
                | ((patch[0, 1] > patch[1, 1]) << 6)  # top
                | ((patch[0, 2] > patch[1, 1]) << 5)  # top right
                | ((patch[1, 2] > patch[1, 1]) << 4)
                | ((patch[2, 2] > patch[1, 1]) << 3)
                | ((patch[2, 1] > patch[1, 1]) << 2)
                | ((patch[2, 0] > patch[1, 1]) << 1)
                | ((patch[1, 0] > patch[1, 1]) << 0)
            )
            lbp[value] += 1

    return lbp


def gamma_correction(image: np.ndarray, c, gamma):
    """
    Apply gamma correction to an image.

    Args:
        image (np.ndarray): The input image.
        c: A constant factor to scale the image.
        gamma: The value of gamma for correction.

    Returns:
        np.ndarray: The gamma-corrected image.
    """
    return c * (image.copy() ** gamma)


def histogram_equalization(image: np.ndarray):
    """
    Calculate the histogram equalization of an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The histogram equalized image as a NumPy array.
    """
    histogram = np.histogram(image.flatten(), bins=256, range=[0, 256])[0]

    H_c = np.cumsum(histogram)

    max = np.max(image)
    q = np.array(
        [round((max - 1) * val / (image.shape[0] * image.shape[1])) for val in H_c]
    )
    copy = q[image.flatten()].reshape(image.shape)
    return copy


def median_filter(image, window_height, window_width):
    """
    Apply a median filter to an image.

    Parameters:
        - image: NumPy array representing the input image.
        - window_height: Integer representing the height of the filter window.
        - window_width: Integer representing the width of the filter window.

    Returns:
        - newimage: NumPy array representing the filtered image.
    """
    edgex = window_width // 2
    edgey = window_height // 2
    height = image.shape[0]
    width = image.shape[1]

    newimage = np.zeros((height, width))

    for x in range(edgex, width - edgex):
        for y in range(edgey, height - edgey):
            temp = np.ndarray((window_height, window_width))
            for fx in range(window_width):
                for fy in range(window_height):
                    temp[fy][fx] = image[y + fy - edgey, x + fx - edgex]
            temp.sort()
            med = np.median(temp)
            newimage[y, x] = med

    return newimage


def grayscale(img):
    """
    Convert an image to grayscale.

    Parameters:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray
