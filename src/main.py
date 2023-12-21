import argparse
import datetime
import json
import logging

import debug_logger.debug_logger as debug_logger
from utils.utils import create_directory


def main():
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--debug", action="store_true")
    args.add_argument("-c", "--config", default="src//config/logging_config.json")

    # Extract arguments
    args = args.parse_args()
    debug = args.debug
    config = args.config

    # Load json config
    config_json = json.load(open(config, "r"))

    # Create log directory
    create_directory(config_json["log_path"])

    # Create log path
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d %H %M %S") + ".log"
    logger = debug_logger.DebugLogger(
        "my_logger",
        level=logging.DEBUG if debug else logging.INFO,
        log_file_name=config_json["log_path"] + "/" + log_filename
        if config_json["save_log_to_file"]
        else None,
    )
    main()
