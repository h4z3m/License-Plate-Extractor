import xml.etree.ElementTree as ET
import requests

from urllib.parse import urlparse
import random
import urllib.request


def extract_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    i = 0
    for plate in root.findall("plate"):
        country = plate.find("country").text
        car = plate.find("car").text
        model = plate.find("model").text
        model2 = plate.find("model2").text
        photo_url = plate.find("photo_url").text
        plate_number = plate.find("plate_number").text
        plate_number_image_url = plate.find("plate_number_image_url").text
        plate_id = plate.find("plate_id").text
        plate_title = plate.find("plate_title").text
        plate_region = plate.find("plate_region").text
        fon_id = plate.find("fon_id").text
        fon_title = plate.find("fon_title").text
        tags = plate.find("tags").text
        link = plate.find("link").text

        plate_data = {
            "country": country,
            "car": car,
            "model": model,
            "model2": model2,
            "photo_url": photo_url,
            "plate_number": plate_number,
            "plate_number_image_url": plate_number_image_url,
            "plate_id": plate_id,
            "plate_title": plate_title,
            "plate_region": plate_region,
            "fon_id": fon_id,
            "fon_title": fon_title,
            "tags": tags,
            "link": link,
        }

        def download_image(url):
            opener = urllib.request.build_opener()
            opener.addheaders = [
                (
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36",
                )
            ]
            photo_file_name = f"{i}_photo"

            urllib.request.install_opener(opener)
            fullname = "./data/xml_dataset/" + photo_file_name + ".jpg"
            urllib.request.urlretrieve(url, fullname)

        download_image(photo_url)
        i += 1

    return data


# Usage example
xml_file = "./data/exp_eg_001.xml"
result = extract_data_from_xml(xml_file)
