import cv2
import numpy as np

from image_management.ImageTransformerBase import ImageTransformerBase


class ImageTransformerDepth(ImageTransformerBase):
    @staticmethod
    def load(image_path):
        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def save(image, output_path):
        image = ImageTransformerDepth.normalize(image=image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = ImageTransformerDepth.transform_dtype(image=image, dtype=np.uint8)
        ImageTransformerBase.save(image=image, output_path=output_path)

    @staticmethod
    def remove_data_between_distance_neighbors(image, min_depth, max_depth, other_image=None, kernel_shape=(3, 3),
                                               iterations=10):
        mask_out_of_range = (image > max_depth) | (image < min_depth)
        mask_out_of_range = mask_out_of_range.astype(np.uint8)

        kernel = np.ones(kernel_shape, np.uint8)
        mask_with_neighbors = cv2.dilate(mask_out_of_range, kernel, iterations=iterations)

        image = ImageTransformerDepth.apply_mask(image=image, condition=mask_with_neighbors == 1,
                                                 value=other_image if other_image is not None else 0)

        return image, mask_with_neighbors

    @staticmethod
    def remove_data_between_distance(image, min_depth, max_depth):
        mask_out_of_range = (image > max_depth) | (image < min_depth)
        mask_out_of_range = mask_out_of_range.astype(np.uint8)

        return np.where(mask_out_of_range == 1, 0, image)

    @staticmethod
    def normalize_between_distance(image, min_depth, max_depth):
        return ((image - min_depth) / (max_depth - min_depth)) * 255.0

    @staticmethod
    def set_data_between_distance(image, min_depth, max_depth):
        return np.clip(image, min_depth, max_depth)

    @staticmethod
    def invert(image):
        image = ImageTransformerDepth.normalize(image=image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = ImageTransformerDepth.transform_dtype(image=image, dtype=np.uint8)
        return 255 - image
