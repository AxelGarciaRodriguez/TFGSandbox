import argparse
import logging
import sys
from copy import deepcopy

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

from image_manager.ImageProcessorDepth import ImageProcessorDepth
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectController
from literals import KinectFrames


def get_args():
    parser = argparse.ArgumentParser(prog="app",
                                     description='Principal application',
                                     epilog='',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--logging', help='Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40 or CRITICAL=50)',
                        type=int, required=False, default=logging.INFO)

    args, unknown = parser.parse_known_args()
    return args


def principal_application(principal_screen):
    pass


def projector_application(projector_screen, kinect):
    # INSTANTIATE PROJECTOR APP
    kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH)
    kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)

    min_depth, max_depth = kinect.kinect_calibrations[KinectFrames.DEPTH.name].get_depth()

    previous_depth = None
    # PROJECTOR
    projector_screen.create_window_calibrate(window_name="Projector Window",
                                             image=kinect_depth_processor.image, fullscreen=True)
    while projector_screen.check_if_window_active(window_name="Projector Window"):
        kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH)
        if kinect_image_depth is not None:
            kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)
            # REMOVE SENSOR ERRORS
            kinect_depth_processor.remove_zeros()

            # REMOVE DATA NOT IN RANGE
            kinect_depth_processor.remove_data_between_distance_option_b(min_depth=min_depth, max_depth=max_depth)

            # COMBINE WITH PREVIOUS IMAGE
            if previous_depth is not None:
                mask_zeros = kinect_depth_processor.image == 0
                kinect_depth_processor.image[mask_zeros] = previous_depth[mask_zeros]

                umbral = 5
                mask_noise = (~mask_zeros) & (np.abs(kinect_depth_processor.image - previous_depth) < umbral)
                kinect_depth_processor.image[mask_noise] = previous_depth[mask_noise]

            previous_depth = deepcopy(kinect_depth_processor.image)

            # NORMALIZE IMAGE
            img_depth_normalized = ((kinect_depth_processor.image - min_depth) / (max_depth - min_depth)) * 255.0
            kinect_depth_processor.update(image=img_depth_normalized)

            # TRANSFORM IMAGE TO 8UNIT
            kinect_depth_processor.transform_dtype()

            # APPLY GAUSSIAN FILTER
            kinect_depth_processor.degaussing()

            # # INVERT DATA TO GENERATE INVERSE COLOR MAP
            kinect_depth_processor.invert()

            # APPLY COLORMAP
            kinect_depth_processor.apply_colormap()

            projector_screen.update_window_image_calibrate(window_name="Projector Window",
                                                           image=kinect_depth_processor.image)


def main():
    args = get_args()

    # Initialize logging class
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                        level=args.logging,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%H:%M:%S')

    # Initialize Kinect, Principal Screen and Projector Screen
    try:
        principal_screen, projector_screen = selector_screens()
        kinect = KinectController(kinect_frames=[KinectFrames.DEPTH])
    except Exception as error:
        logging.error(f"Error trying to instantiate screens/kinect: {error}")
        raise error

    principal_application(principal_screen=principal_screen)
    projector_application(projector_screen=projector_screen, kinect=kinect)


if __name__ == '__main__':
    main()
