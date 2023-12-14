import numpy as np
import easyocr
import argparse
import cv2
import os


def convert(bx, by, bw, bh, w, h):
    xmx = (2 * w * bx + w * bw) / 2
    xmn = 2 * w * bx - xmx
    ymx = (2 * h * by + h * bh) / 2
    ymn = 2 * h * by - ymx
    return xmn, xmx, ymn, ymx


def benchmark(image, roi: np.ndarray, labelBoundingBox: str):
    easyocr_reader = easyocr.Reader(["ar"])
    results = easyocr_reader.readtext(roi)

    # Cut out label bounding box
    h = image.shape[0]
    w = image.shape[1]
    # Convert string to float
    labelBoundingBox = [float(i) for i in labelBoundingBox]

    xmn, xmx, ymn, ymx = convert(*labelBoundingBox[1:], w, h)

    def rint(x):
        return int(round(x))

    labelArea = image[rint(ymn) : rint(ymx), rint(xmn) : rint(xmx)]
    # TODO Preprocess label area
    new_image = 0
    results_label = easyocr_reader.readtext(new_image)
    print("Results: ", results)
    print("Results Label: ", results_label)
    return results[0][1] == results_label[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-d", type=str)
    parser.add_argument("--dataset-labels", "-l", type=str)
    args = parser.parse_args()
    passed = 0
    # Loop on all images in dataset path
    for image_path in os.listdir(args.dataset_path):
        image = cv2.imread(os.path.join(args.dataset_path, image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # TODO Get ROI
        roi = 0

        # Compare with labels
        label = open(
            os.path.join(args.dataset_labels, image_path.split(".")[0] + ".txt")
        )

        label = label.read()
        if benchmark(image, roi, label.split(" ")):
            passed += 1
            print(f"Passed: {passed}/{len(os.listdir(args.dataset_path))}")
        else:
            print(f"Failed: {image_path}")
            cv2.imshow("image", image)
    # Print accuracy
    print(f"Accuracy: {passed}/{len(os.listdir(args.dataset_path))}")
