import argparse
import datetime
import json
import logging

import cv2
import numpy as np

import debug_logger.debug_logger as debug_logger
from utils.utils import create_directory


def main(image_path):
    from license_plate_extractor import LicensePlateExtractor

    lpe = LicensePlateExtractor(image_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--debug", action="store_true")
    args.add_argument(
        "-c",
        "--config",
        default="src/config/logging_config.json",
        help="Path to logging config file",
    )
    args.add_argument("-i", "--image-path", help="Path to an image")

    # Extract arguments
    args = args.parse_args()
    debug = args.debug
    config = args.config

    # Load json config
    config_json = json.load(open(config, "r"))
    if config_json["save_log_to_file"]:
        # Create log directory
        create_directory(config_json["log_path"])

        # Create log path
        log_filename = datetime.datetime.now().strftime("%Y-%m-%d %H %M %S") + ".log"

    logger = debug_logger.DebugLogger(
        name="logger", level=logging.DEBUG if debug else logging.INFO
    )

    main(image_path=args.image_path)
