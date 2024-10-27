import logging
import time
from functools import reduce
from typing import List

import cv2
import numpy as np

from calibrations.CalibrationFile import CalibrationClass
from kinect_module import PyKinectRuntime

from literals import KINECT_MAX_CHECKS_CONNECTION, KINECT_SECONDS_BETWEEN_CHECK_CONNECTION, KINECT_CALIBRATION_PATH, \
    KINECT_CALIBRATION_FILENAME, KinectFrames
from kinect_controller.KinectLock import lock
from utils import generate_relative_path


class KinectController(object):
    def __init__(self, kinect_frames: List[KinectFrames]):
        logging.info("Initializing Kinect Camera ...")

        # Instantiate values
        self.kinect = None
        self.kinect_frames = kinect_frames

        # Generate kinect frames list
        kinect_frames_values = [kinect_frame.value for kinect_frame in self.kinect_frames]

        # Initiate PyKinect2 runtime
        self.kinect = PyKinectRuntime.PyKinectRuntime(kinect_frames_values[0]) if len(
            kinect_frames_values) == 1 else PyKinectRuntime.PyKinectRuntime(
            reduce(lambda x, y: x | y, kinect_frames_values))

        # Test Connection
        for i in range(KINECT_MAX_CHECKS_CONNECTION):
            time.sleep(KINECT_SECONDS_BETWEEN_CHECK_CONNECTION)
            if self.check_if_new_image(self.kinect_frames[0]):
                break
            logging.warning(
                f"Retrying connection with Kinect camera, {KINECT_MAX_CHECKS_CONNECTION - i - 1} retries available")
        else:
            raise RuntimeError("Cannot detect Kinect Camera")

        # Read calibrations
        self.kinect_calibrations = {}
        for kinect_frame in self.kinect_frames:
            calibration_path = generate_relative_path(
                [KINECT_CALIBRATION_PATH, kinect_frame.name, KINECT_CALIBRATION_FILENAME])
            self.kinect_calibrations[kinect_frame.name] = CalibrationClass(calibration_path_file=calibration_path)
            self.kinect_calibrations[kinect_frame.name].read_calibration()

    # region Get Images
    def check_if_new_image(self, kinect_frame: KinectFrames):
        if kinect_frame == KinectFrames.COLOR:
            return self.kinect.has_new_color_frame()
        elif kinect_frame == KinectFrames.DEPTH:
            return self.kinect.has_new_depth_frame()
        elif kinect_frame == KinectFrames.INFRARED:
            return self.kinect.has_new_infrared_frame()

        raise ValueError(f"Cannot manage kinect frame {kinect_frame.name} in KinectController wrapper")

    def get_frame(self, kinect_frame: KinectFrames):
        if self.check_if_new_image(kinect_frame=kinect_frame):
            with lock:
                if kinect_frame == KinectFrames.COLOR:
                    kinect_frame_obj = self.kinect.get_last_color_frame()
                elif kinect_frame == KinectFrames.DEPTH:
                    kinect_frame_obj = self.kinect.get_last_depth_frame()
                elif kinect_frame == KinectFrames.INFRARED:
                    kinect_frame_obj = self.kinect.get_last_infrared_frame()
                else:
                    raise ValueError(f"Cannot manage kinect frame {kinect_frame.name} in KinectController wrapper")

            return kinect_frame_obj
        else:
            logging.debug("Not found new frame to get in get_frame method")
            time.sleep(0.01)
        return None

    def get_image(self, kinect_frame: KinectFrames):
        frame = self.get_frame(kinect_frame=kinect_frame)
        if frame is None:
            logging.debug("Not found new frame to transform in get_image method")
            return None

        if kinect_frame == KinectFrames.COLOR:
            image = frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(
                np.uint8)
            if image.shape[-1] == 4:
                image = image[..., :3]
        elif kinect_frame == KinectFrames.DEPTH:
            image = frame.reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width)).astype(
                np.uint16)
        elif kinect_frame == KinectFrames.INFRARED:
            image = frame.reshape(
                (self.kinect.infrared_frame_desc.Height, self.kinect.infrared_frame_desc.Width)).astype(np.uint16)
        else:
            raise ValueError(f"Cannot manage kinect image {kinect_frame.name} in KinectController wrapper")

        image = cv2.flip(image, 1)
        return image

    def get_image_calibrate(self, kinect_frame: KinectFrames, avoid_camera_matrix=False, avoid_camera_focus=False):
        image = self.get_image(kinect_frame=kinect_frame)
        if image is not None:
            if kinect_frame.name in self.kinect_calibrations.keys():
                if not avoid_camera_matrix:
                    image = self.apply_camera_calibration(kinect_frame=kinect_frame, image=image)
                if not avoid_camera_focus:
                    image = self.apply_camera_focus(kinect_frame=kinect_frame, image=image)

        return image

    def apply_camera_calibration(self, kinect_frame: KinectFrames, image):
        image = self.kinect_calibrations[kinect_frame.name].applied_camera_calibration(image=image)
        return image

    def apply_camera_focus(self, kinect_frame: KinectFrames, image):
        image = self.kinect_calibrations[kinect_frame.name].applied_camera_focus(image=image)
        return image

    def save_calibrations(self):
        for kinect_frame in self.kinect_frames:
            self.kinect_calibrations[kinect_frame.name].save_calibration()

    # endregion

    # region Kinect Management
    def close(self):
        self.kinect.close()
    # endregion
