import argparse
import datetime
import json
import logging

import cv2
import numpy as np
import csv
import debug_logger.debug_logger as debug_logger
from utils.utils import create_directory


def main(dataset_path, lpe_config_path, pe_config_path, n, op_path):
    from license_plate_extractor import LicensePlateExtractor

    with open("output.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        field = ["image_name", "plate_type"]
        writer.writerow(field)
        LicensePlateExtractor.load_config(lpe_config_path, pe_config_path)
        # Loop on the first 100 images
        for i in range(1200, 1301):
            filename = dataset_path + "/{:04d}.jpg".format(i)
            image, candidates, annotated_image = LicensePlateExtractor.extract_plate(
                filename
            )
            original_image = cv2.imread(filename)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # Save images
            if len(candidates) >= 2:
                cv2.imwrite(op_path + "/{:04d}_0.jpg".format(i), candidates[0][0])
                cv2.imwrite(op_path + "/{:04d}_1.jpg".format(i), candidates[1][0])
                print(*candidates[0][3])
                print(*candidates[1][3])
                candidate_1_type = LicensePlateExtractor.extract_type(
                    original_image, *candidates[0][3]
                )
                candidate_2_type = LicensePlateExtractor.extract_type(
                    original_image, *candidates[1][3]
                )
                logger.debug(filename + " " + candidate_1_type)
                logger.debug(filename + " " + candidate_2_type)

                writer.writerow([filename, candidate_1_type])
                writer.writerow([filename, candidate_2_type])
            elif len(candidates) == 1:
                cv2.imwrite(op_path + "/{:04d}_0.jpg".format(i), candidates[0][0])
                candidate_1_type = LicensePlateExtractor.extract_type(
                    original_image, *candidates[0][3]
                )
                logger.debug(filename + " " + candidate_1_type)
                writer.writerow([filename, candidate_1_type])

            else:
                logger.debug("Error: No candidates found for image #{}".format(i))


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
        "-n",
        "--number",
        default=10,
        type=int,
        help="Number of images to process from dataset path",
    )

    # Extract arguments
    args = args.parse_args()
    debug = args.debug
    config = args.log_config
    dataset_path = args.dataset_path
    n = args.number
    lpe_config_path = args.extractor_config
    pe_config_path = args.plate_config
    op_path = args.output_path
    # Load json config
    config_json = json.load(open(config, "r"))
    if config_json["save_log_to_file"]:
        # Create log directory
        create_directory(config_json["log_path"])
        # Create log path
        log_filename = datetime.datetime.now().strftime("%Y-%m-%d %H %M %S") + ".log"

    logger = debug_logger.DebugLogger(
        name="logger", level=logging.DEBUG if debug else logging.ERROR
    )

    main(dataset_path, lpe_config_path, pe_config_path, n, op_path)
