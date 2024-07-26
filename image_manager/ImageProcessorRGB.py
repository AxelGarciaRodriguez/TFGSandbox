import cv2

from image_manager.ImageProcessor import ImageProcessor


class ImageProcessorRGB(ImageProcessor):
    def __init__(self, image=None, image_absolute_path=None):
        super().__init__(image=image, image_absolute_path=image_absolute_path)
