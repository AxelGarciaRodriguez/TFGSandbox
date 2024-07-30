import cv2
import numpy as np

from image_manager.ImageProcessor import ImageProcessor


class ImageProcessorIR(ImageProcessor):
    def __init__(self, image=None, image_absolute_path=None):
        super().__init__(image=image, image_absolute_path=image_absolute_path)

    def load(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.update(image=image)

    def save(self, output_path):
        self.normalize()
        self.transform_dtype(dtype=np.uint8)

        # jpg_frame = np.zeros((self.image_shape[0], self.image_shape[1], 3), np.uint8)
        # jpg_frame[:, :, 0] = self.image
        # jpg_frame[:, :, 1] = self.image
        # jpg_frame[:, :, 2] = self.image
        #
        # self.update(image=jpg_frame)
        super().save(output_path=output_path)
        self.restore()

    def find_chessboard_corners(self, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE):
        self.normalize()
        self.transform_dtype(dtype=np.uint8)

        ret, corners = cv2.findChessboardCorners(image=self.image, patternSize=pattern_size, flags=flags)

        self.restore()
        return ret, corners

    def calculate_sub_pix_corner(self, corners,
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        self.normalize()
        self.transform_dtype(dtype=np.uint8)

        corners = cv2.cornerSubPix(self.image, corners, (5, 5), (-1, -1), criteria)
        self.restore()
        return corners
