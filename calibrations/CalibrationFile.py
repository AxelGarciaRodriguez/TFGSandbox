import logging
import os

import cv2
import numpy as np

from literals import CAMERA_CALIBRATION_VARIABLE, CAMERA_DISTORTION_VARIABLE, CAMERA_ROTATION_VARIABLE, \
    CAMERA_TRANSLATION_VARIABLE, OBJ_POINTS_KEY, IMG_POINTS_KEY, FOCUS_HOMOGRAPHY_VARIABLE, \
    FOCUS_INV_HOMOGRAPHY_VARIABLE, FOCUS_CORDS_VARIABLE, FOCUS_CORDS_ORIGINAL_VARIABLE, \
    FOCUS_DIMENSION_ORIGINAL_VARIABLE, MIN_DEPTH_VARIABLE, MAX_DEPTH_VARIABLE, STANDARD_MIN_DEPTH, STANDARD_MAX_DEPTH
from utils import generate_folders, ordering_points


class CalibrationClass:
    def __init__(self, cords=None, original_cords=None, matrix_homography=None,
                 matrix_inverse_homography=None, obj_points=None, img_points=None, camera_matrix=None,
                 cof_distortion=None, camera_rotation=None, camera_translation=None, calibration_path_file=None):
        # FOCUS
        self.cords = cords
        self.original_cords = original_cords
        self.matrix_homography = matrix_homography
        self.matrix_inverse_homography = matrix_inverse_homography

        # DEPTH (mm)
        self.min_depth = None
        self.max_depth = None

        # CALIBRATION
        self.obj_points = obj_points
        self.img_points = img_points
        self.camera_matrix = camera_matrix
        self.cof_distortion = cof_distortion
        self.camera_rotation = camera_rotation
        self.camera_translation = camera_translation

        # FILES
        self.calibration_path_file = calibration_path_file

    def set_calibrations(self, calibration):
        if CAMERA_CALIBRATION_VARIABLE in calibration.keys():
            self.camera_matrix = calibration[CAMERA_CALIBRATION_VARIABLE]
        if CAMERA_DISTORTION_VARIABLE in calibration.keys():
            self.cof_distortion = calibration[CAMERA_DISTORTION_VARIABLE]
        if CAMERA_ROTATION_VARIABLE in calibration.keys():
            self.camera_rotation = calibration[CAMERA_ROTATION_VARIABLE]
        if CAMERA_TRANSLATION_VARIABLE in calibration.keys():
            self.camera_translation = calibration[CAMERA_TRANSLATION_VARIABLE]
        if OBJ_POINTS_KEY in calibration.keys():
            self.obj_points = calibration[OBJ_POINTS_KEY]
        if IMG_POINTS_KEY in calibration.keys():
            self.img_points = calibration[IMG_POINTS_KEY]

        if FOCUS_HOMOGRAPHY_VARIABLE in calibration.keys():
            self.matrix_homography = calibration[FOCUS_HOMOGRAPHY_VARIABLE]
        if FOCUS_INV_HOMOGRAPHY_VARIABLE in calibration.keys():
            self.matrix_inverse_homography = calibration[FOCUS_INV_HOMOGRAPHY_VARIABLE]
        if FOCUS_CORDS_VARIABLE in calibration.keys():
            self.cords = calibration[FOCUS_CORDS_VARIABLE]
        if FOCUS_CORDS_ORIGINAL_VARIABLE in calibration.keys():
            self.original_cords = calibration[FOCUS_CORDS_ORIGINAL_VARIABLE]
        if MIN_DEPTH_VARIABLE in calibration.keys():
            self.min_depth = int(calibration[MIN_DEPTH_VARIABLE])
        if MAX_DEPTH_VARIABLE in calibration.keys():
            self.max_depth = int(calibration[MAX_DEPTH_VARIABLE])

    def read_calibration(self, calibration_path_file=None):
        if calibration_path_file:
            self.calibration_path_file = calibration_path_file

        if not self.calibration_path_file:
            raise FileNotFoundError("Missing calibration path file to load")

        if os.path.exists(self.calibration_path_file) and os.path.isfile(self.calibration_path_file):
            try:
                calibration = np.load(self.calibration_path_file)
                self.set_calibrations(calibration=calibration)

            except Exception as error:
                logging.error(f"Kinect error loading file {self.calibration_path_file}: {error}")
        else:
            logging.warning("Not found any calibration files for Kinect, please calibrate camera.")

    def save_calibration(self, calibration_path_file=None):
        if calibration_path_file:
            self.calibration_path_file = calibration_path_file

        if not self.calibration_path_file:
            raise FileNotFoundError("Missing calibration path file to save")

        arguments_saved = {}

        if self.camera_matrix is not None:
            arguments_saved[CAMERA_CALIBRATION_VARIABLE] = self.camera_matrix
        if self.cof_distortion is not None:
            arguments_saved[CAMERA_DISTORTION_VARIABLE] = self.cof_distortion
        if self.camera_rotation is not None:
            arguments_saved[CAMERA_ROTATION_VARIABLE] = self.camera_rotation
        if self.camera_translation is not None:
            arguments_saved[CAMERA_TRANSLATION_VARIABLE] = self.camera_translation
        if self.obj_points is not None:
            arguments_saved[OBJ_POINTS_KEY] = self.obj_points
        if self.img_points is not None:
            arguments_saved[IMG_POINTS_KEY] = self.img_points

        if self.matrix_homography is not None:
            arguments_saved[FOCUS_HOMOGRAPHY_VARIABLE] = self.matrix_homography
        if self.matrix_inverse_homography is not None:
            arguments_saved[FOCUS_INV_HOMOGRAPHY_VARIABLE] = self.matrix_inverse_homography
        if self.cords is not None:
            arguments_saved[FOCUS_CORDS_VARIABLE] = self.cords
        if self.original_cords is not None:
            arguments_saved[FOCUS_CORDS_ORIGINAL_VARIABLE] = self.original_cords
        if self.min_depth is not None:
            arguments_saved[MIN_DEPTH_VARIABLE] = self.min_depth
        if self.max_depth is not None:
            arguments_saved[MAX_DEPTH_VARIABLE] = self.max_depth

        # Save calibrate file
        generate_folders(self.calibration_path_file)
        np.savez(self.calibration_path_file, **arguments_saved)

    def calculate_inverse_homography(self, matrix_homography=None):
        if matrix_homography is not None:
            self.matrix_homography = matrix_homography

        if self.matrix_homography is None:
            logging.warning("Cannot calculate inverse homography, missing homography matrix")
            return

        success, homography_inverse = cv2.invert(self.matrix_homography)

        if success:
            self.matrix_inverse_homography = homography_inverse
        else:
            logging.error("Cannot calculate inverse homography, error matrix")

    def calculate_homography(self, cords=None, original_cords=None):
        if cords is not None:
            self.cords = cords
        if original_cords is not None:
            self.original_cords = original_cords

        if self.cords is None or self.original_cords is None:
            logging.warning("Cannot calculate homography, missing cords")
            return

        # ORDER CORDS
        cords = np.array(self.cords, dtype=np.float32)
        original_cords = np.array(self.original_cords, dtype=np.float32)

        cords_ordered = np.float32(ordering_points(original_cords, cords))
        self.matrix_homography = cv2.getPerspectiveTransform(cords_ordered, original_cords)
        self.calculate_inverse_homography()

    def applied_camera_calibration(self, image):
        if self.camera_matrix is not None and self.cof_distortion is not None:
            image = cv2.undistort(image, self.camera_matrix, self.cof_distortion)
        return image

    @staticmethod
    def _applied_camera_matrix(image, matrix, output_size=None):
        if matrix is not None:
            if not output_size:
                height, width = image.shape[:2]
                output_size = (width, height)
            else:
                image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)

            image = cv2.warpPerspective(image, matrix, output_size)

        return image

    def applied_camera_focus(self, image, output_size=None):
        return self._applied_camera_matrix(image=image, matrix=self.matrix_homography, output_size=output_size)

    def applied_inverse_camera_focus(self, image, output_size=None):
        return self._applied_camera_matrix(image=image, matrix=self.matrix_inverse_homography, output_size=output_size)

    def get_depth(self):
        min_depth = self.min_depth
        max_depth = self.max_depth
        if min_depth is None:
            logging.warning(f"NOT MIN DEPTH DEFINED, GETTING STANDARD MIN DEPTH {STANDARD_MIN_DEPTH}")
            min_depth = STANDARD_MIN_DEPTH
        if max_depth is None:
            logging.warning(f"NOT MAX DEPTH DEFINED, GETTING STANDARD MAX DEPTH {STANDARD_MAX_DEPTH}")
            max_depth = STANDARD_MAX_DEPTH
        return min_depth, max_depth
