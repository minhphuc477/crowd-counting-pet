from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb_image(path):
    with Image.open(path) as img:
        return img.convert('RGB')
