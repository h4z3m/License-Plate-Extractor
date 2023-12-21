from debug_logger.debug_logger import DebugLogger
from plate_extraction.plate_extraction import extract_color_region

logger = DebugLogger(name="logger")


class LicensePlateExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        pass

    def extract_plate(self, image):
        pass

    def extract_region(self, image):
        pass

    def get_plate_number(self, image):
        pass

    def get_plate_region(self, image):
        pass
