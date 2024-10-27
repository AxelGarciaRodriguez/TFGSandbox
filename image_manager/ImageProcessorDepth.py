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

    def remove_data_between_distance_option_a(self, min_depth, max_depth, other_image=None):
        mask_out_of_range = (self.image > max_depth) | (self.image < min_depth)
        mask_out_of_range = mask_out_of_range.astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask_with_neighbors = cv2.dilate(mask_out_of_range, kernel, iterations=10)

        if other_image is None:
            # image = np.clip(self.image, min_depth, max_depth)
            image = np.where(mask_with_neighbors == 1, 0, self.image)
        else:
            image = np.where(mask_with_neighbors == 1, other_image, self.image)

        self.update(image=image)
        return image, mask_with_neighbors

    def remove_data_between_distance_option_b(self, min_depth, max_depth, other_image=None):
        mask_out_of_range = (self.image > max_depth) | (self.image < min_depth)
        mask_out_of_range = mask_out_of_range.astype(np.uint8)

        # kernel = np.ones((3, 3), np.uint8)
        # mask_with_neighbors = cv2.dilate(mask_out_of_range, kernel, iterations=30)

        image = np.where(mask_out_of_range == 1, 0, self.image)
        self.update(image=image)
        return image

    def remove_data_between_distance_option_c(self, min_depth, max_depth, other_image=None):
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
