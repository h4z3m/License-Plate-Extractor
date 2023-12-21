import os
from matplotlib import pyplot as plt


def create_directory(path):
    """
    Create a directory if it does not already exist.

    Args:
        path (str): The path of the directory to be created.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.mkdir(path)


def plot_side_by_side_images(img1, img2, title1="Original", title2="Compared"):
    """
    Plots two images side by side with titles.

    Args:
        img1: The first image to be plotted.
        img2: The second image to be plotted.
        title1: The title for the first image. Defaults to "Original".
        title2: The title for the second image. Defaults to "Compared".

    Returns:
        None
    """
    plt.figure(figsize=[7, 7])

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = original image
    plt.imshow(img1, cmap="gray")
    plt.title(title1)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = gray image
    plt.imshow(img2, cmap="gray")
    plt.title(title2)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()


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
