import logging, sys


class DebugLogger(logging.Logger):
    def __init__(
        self, name: str = None, level: int = logging.DEBUG, log_file_name=None
    ):
        super().__init__(name, level)

        fmt = "%(asctime)s %(filename)-10s %(levelname)-3s: %(message)s"
        fmt_date = "%T"
        formatter = logging.Formatter(fmt=fmt, datefmt=fmt_date)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        # Set the logging format with colors
        logging.addLevelName(
            logging.WARNING,
            "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING),
        )
        logging.addLevelName(
            logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR)
        )
        # Add the handlers to the logger
        self.addHandler(console_handler)

        # Optionally, add a file handler
        if log_file_name:
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
