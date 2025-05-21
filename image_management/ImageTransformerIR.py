import cv2
import numpy as np

from image_management.ImageTransformerBase import ImageTransformerBase


class ImageTransformerIR(ImageTransformerBase):

    @staticmethod
    def load(image_path):
        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def save(image, output_path):
        image = ImageTransformerIR.normalize(image=image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = ImageTransformerIR.transform_dtype(image=image, dtype=np.uint8)
        ImageTransformerBase.save(image=image, output_path=output_path)

    @staticmethod
    def find_chessboard_corners(image, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE):
        image_normalized = ImageTransformerIR.normalize(image=image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_uint8 = ImageTransformerIR.transform_dtype(image=image_normalized, dtype=np.uint8)

        ret, corners = cv2.findChessboardCorners(image=image_uint8, patternSize=pattern_size, flags=flags)
        return ret, corners

    @staticmethod
    def calculate_sub_pix_corner(image, corners,
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        image_normalized = ImageTransformerIR.normalize(image=image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_uint8 = ImageTransformerIR.transform_dtype(image=image_normalized, dtype=np.uint8)

        corners = cv2.cornerSubPix(image_uint8, corners, (5, 5), (-1, -1), criteria)
        return corners
