import cv2
import numpy as np

from image_manager.ImageProcessor import ImageProcessor


class ImageProcessorDepth(ImageProcessor):
    def __init__(self, image=None, image_absolute_path=None):
        super().__init__(image=image, image_absolute_path=image_absolute_path)

    def load(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.update(image=image)

    def save(self, output_path):
        self.restore()
        self.normalize()
        self.transform_dtype(dtype=np.uint8)
        super().save(output_path=output_path)
        self.restore()

    def remove_data_between_distance(self, min_depth, max_depth):
        image = np.clip(self.image, min_depth, max_depth)
        self.update(image=image)
        return image

    def invert(self):
        image = self.image
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = np.uint8(image)

        image = 255 - image
        self.update(image=image)
        return self.image
