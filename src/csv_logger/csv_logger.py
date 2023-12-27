import csv
import logging


class DictCSVLogFormatter(logging.Formatter):
    def __init__(self, fieldnames, filename):
        super().__init__()
        self.fieldnames = fieldnames
        self.filename = filename

    def format(self, record):
        """Formats a given record into a CSV row using the fieldnames provided in the class

        Args:
            record (dict): Record dictionary with fieldnames as keys and their values.

        Returns:
            str:    Empty string
        """
        # Convert the log record message (dictionary) into a DictWriter-compatible dictionary
        log_dict = record.msg

        # Create a CSV writer object
        csv_file = open(self.filename, "a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)

        # Write header if it doesn't exist
        if csv_file.tell() == 0:
            csv_writer.writeheader()

        # Write the log entry as a row in the CSV file
        try:
            csv_writer.writerow(log_dict)
        except Exception:
            pass
        finally:
            # Close the CSV file
            csv_file.close()

        return ""


def setupLogger(
    logger_name: str, fieldnames: list[str], filename: str
) -> logging.Logger:
    """
    Sets up a logger with a custom log handler.

    Parameters:
    - fieldnames (list[str]): A list of field names for the log formatter.
    - filename (str): The name of the file to log to.

    Returns:
    - None
    """
    # Add the custom log handler to the root logger
    csv_handler = logging.StreamHandler()
    csv_handler.setFormatter(DictCSVLogFormatter(fieldnames, filename))
    csv_logger = logging.getLogger(logger_name)
    csv_logger.setLevel(logging.DEBUG)
    csv_logger.addHandler(csv_handler)
    return csv_logger
