import argparse
import csv
import datetime
import json
import logging
from typing import Tuple

import cv2
import numpy as np

import debug_logger.debug_logger as debug_logger
from csv_logger.csv_logger import setupLogger
from ocr.ocr import OCR
from segmentation.segmentation import Segmentation
from utils.utils import create_directory, debug_plot_image, plot_image


def main(
    dataset_path: str,
    lpe_config_path: str,
    pe_config_path: str,
    image_range: Tuple[int, int],
    op_path: str,
    ocr_config_path: str,
):
    from license_plate_extractor import LicensePlateExtractor

    # Create a CSV logger with the fieldnames
    csv_logger = setupLogger(
        "csv_logger",
        [
            "Picture",
            "Extracted_picture",
            "OCR_text",
            "Plate_type",
        ],
        f"output_{image_range[0]}_to_{image_range[1]}.csv",
    )

    def save_image_data(idx, sub_idx, op_path, filename, candidate, original_image):
        cv2.imwrite(f"{op_path}/{idx:04d}_{sub_idx}.jpg", candidate[0])
        type = LicensePlateExtractor.extract_type(original_image, *candidate[3])
        debug_plot_image(candidate[0])
        ocr_text = LicensePlateExtractor.get_plate_number(original_image, candidate[0])
        # TODO annotate here
        csv_logger.info(
            {
                "Picture": filename,
                "Extracted_picture": f"{idx:04d}_{sub_idx}.jpg",
                "OCR_text": ocr_text,
                "Plate_type": type,
            }
        )
        # TODO Save the annotated image

    LicensePlateExtractor.load_config(lpe_config_path, pe_config_path, ocr_config_path)

    for i in range(*image_range):
        filename = f"{dataset_path}/{i:04d}.jpg"
        image, candidates, annotated_image = LicensePlateExtractor.extract_plate(
            filename
        )
        original_image = cv2.imread(filename)
        if original_image is None:
            logger.error(f"Could not load image: {filename}")
            exit(-1)

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if len(candidates) >= 2:
            save_image_data(i, 0, op_path, filename, candidates[0], original_image)
            save_image_data(i, 1, op_path, filename, candidates[1], original_image)
        elif len(candidates) == 1:
            save_image_data(i, 0, op_path, filename, candidates[0], original_image)
        else:
            logger.error(f"No candidates found for image #{i}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("-d", "--debug", action="store_true")

    args.add_argument(
        "-lc",
        "--log-config",
        default="src/config/logging_config.json",
        help="Path to logging config file",
    )
    args.add_argument(
        "-pc",
        "--plate-config",
        default="src/config/plate_extraction_config.json",
        help="Path to plate extraction config file",
    )

    args.add_argument(
        "-ec",
        "--extractor-config",
        default="src/config/lpe_config.json",
        help="Path to license plate extractor config file",
    )
    args.add_argument(
        "-dp",
        "--dataset-path",
        default="../data/Vehicles",
        help="Path to the dataset folder",
    )
    args.add_argument(
        "-o",
        "--output-path",
        default="../data/output",
        help="Path to the output folder",
    )
    args.add_argument(
        "-r",
        "--range",
        default=[1, 10],
        type=int,
        nargs=2,
        help="Range of images to process from dataset path",
    )

    args.add_argument(
        "-oc",
        "--ocr-config",
        default="src/config/ocr_config.json",
        help="Path to OCR config file",
    )

    # Extract arguments
    args = args.parse_args()
    debug = args.debug
    config = args.log_config
    dataset_path = args.dataset_path
    r = args.range
    lpe_config_path = args.extractor_config
    pe_config_path = args.plate_config
    op_path = args.output_path
    ocr_config_path = args.ocr_config

    create_directory(op_path)  # Create output path if it does not exist
    create_directory(op_path + "/annotated")

    # Load json config
    config_json = json.load(open(config, "r"))
    if config_json["save_log_to_file"]:
        # Create log directory
        create_directory(config_json["log_path"])
        # Create log path
        log_filename = datetime.datetime.now().strftime("%Y-%m-%d %H %M %S") + ".log"

    # Initialize logger
    logger = debug_logger.DebugLogger(
        name="logger", level=logging.DEBUG if debug else logging.ERROR
    )

    main(dataset_path, lpe_config_path, pe_config_path, r, op_path, ocr_config_path)
