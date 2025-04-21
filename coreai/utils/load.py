import requests
from io import BytesIO
from PIL import Image
import base64


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image_data = base64.b64decode(image_file)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    return image
