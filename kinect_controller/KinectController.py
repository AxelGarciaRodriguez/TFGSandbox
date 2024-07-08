import enum
import logging
import time
from functools import reduce
from typing import List

import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

from literals import KINECT_MAX_CHECKS_CONNECTION, KINECT_SECONDS_BETWEEN_CHECK_CONNECTION
from kinect_controller.KinectLock import lock


class KinectFrames(enum.Enum):
    COLOR = PyKinectV2.FrameSourceTypes_Color
    DEPTH = PyKinectV2.FrameSourceTypes_Depth


class KinectController(object):
    def __init__(self, kinect_frames: List[KinectFrames]):
        logging.info("Initializing Kinect Camera ...")

        # Generate kinect frames list
        kinect_frames_values = [kinect_frame.value for kinect_frame in kinect_frames]

        # Initiate PyKinect2 runtime
        self.kinect = PyKinectRuntime(kinect_frames_values[0]) if len(kinect_frames_values) == 1 else PyKinectRuntime(
            reduce(lambda x, y: x | y, kinect_frames_values))

        # Test Connection
        for i in range(KINECT_MAX_CHECKS_CONNECTION):
            time.sleep(KINECT_SECONDS_BETWEEN_CHECK_CONNECTION)
            if self.check_if_new_image(kinect_frames[0]):
                break
            logging.info(
                f"Retrying connection with Kinect camera, {KINECT_MAX_CHECKS_CONNECTION - i - 1} retries available")
        else:
            raise RuntimeError("Cannot detect Kinect Camera")

    # region Get Images
    def check_if_new_image(self, kinect_frame: KinectFrames):
        if kinect_frame == KinectFrames.COLOR:
            return self.kinect.has_new_color_frame()
        elif kinect_frame == KinectFrames.DEPTH:
            return self.kinect.has_new_depth_frame()

        raise ValueError(f"Cannot manage kinect frame {kinect_frame.name} in KinectController wrapper")

    def get_frame(self, kinect_frame: KinectFrames):
        if self.check_if_new_image(kinect_frame=kinect_frame):
            with lock:
                if kinect_frame == KinectFrames.COLOR:
                    kinect_frame_obj = self.kinect.get_last_color_frame()
                elif kinect_frame == KinectFrames.DEPTH:
                    kinect_frame_obj = self.kinect.get_last_depth_frame()
                else:
                    raise ValueError(f"Cannot manage kinect frame {kinect_frame.name} in KinectController wrapper")

            return kinect_frame_obj
        else:
            logging.debug("Not found new frame to get in get_frame method")
            return None

    def get_image(self, kinect_frame: KinectFrames):
        frame = self.get_frame(kinect_frame=kinect_frame)
        if frame is None:
            logging.debug("Not found new frame to transform in get_image method")
            return None

        if kinect_frame == KinectFrames.COLOR:
            image = frame.reshape((self.kinect.color_frame_desc.Height, self.kinect.color_frame_desc.Width, 4)).astype(
                np.uint8)
        elif kinect_frame == KinectFrames.DEPTH:
            image = frame.reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width)).astype(
                np.uint16)
        else:
            raise ValueError(f"Cannot manage kinect image {kinect_frame.name} in KinectController wrapper")

        return image

    # endregion
