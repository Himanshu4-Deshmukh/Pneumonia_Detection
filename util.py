import base64
from io import BytesIO
from PIL import Image

def base64_to_pil(img_base64):
    image_data = base64.b64decode(img_base64)
    image = Image.open(BytesIO(image_data))
    return image
