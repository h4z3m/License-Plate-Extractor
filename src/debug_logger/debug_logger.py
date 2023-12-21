import logging


class DebugLogger(logging.Logger):
    def __init__(
        self, name: str = None, level: int = logging.DEBUG, log_file_name=None
    ):
        super().__init__(name)
        self.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        fmt = "%(asctime)s %(filename)-10s %(levelname)-3s: %(message)s"
        fmt_date = "%T"
        formatter = logging.Formatter(fmt=fmt, datefmt=fmt_date)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.addHandler(console_handler)

        # Optionally, add a file handler
        if log_file_name:
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
