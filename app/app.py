import argparse
import logging
import sys
from copy import deepcopy

import cv2
import numpy as np

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


def generate_topographic_map(normalized_depth):
    colormap = cv2.applyColorMap(np.uint8(normalized_depth), cv2.COLORMAP_JET)

    return colormap


def projector_application(projector_screen, kinect):
    # INSTANTIATE PROJECTOR APP
    kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH)
    kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)

    min_depth = 1159
    max_depth = 1304

    previous_depth = None

    # PROJECTOR
    projector_screen.create_window_calibrate(window_name="Test projected Image DEPTH",
                                             image=kinect_depth_processor.image, fullscreen=True)
    while projector_screen.check_if_window_active(window_name="Test projected Image DEPTH"):
        kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH)
        if kinect_image_depth is not None:
            kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)
            # REMOVE SENSOR ERRORS
            kinect_depth_processor.remove_zeros()

            # COMBINE WITH PREVIOUS IMAGE
            combined_img = None
            if previous_depth is not None:
                combined_img = (kinect_depth_processor.image + previous_depth) / 2.0

            previous_depth = deepcopy(kinect_depth_processor.image)
            if combined_img is not None:
                kinect_depth_processor.update(image=combined_img)

            # APPLY GAUSSIAN FILTER
            kinect_depth_processor.degaussing()

            # REMOVE DATA NOT IN DISTANCE
            kinect_depth_processor.remove_data_between_distance(min_depth=min_depth, max_depth=max_depth)

            # NORMALIZE IMAGE
            img_depth_normalized = ((kinect_depth_processor.image - min_depth) / (max_depth - min_depth)) * 255.0
            kinect_depth_processor.update(image=img_depth_normalized)

            # TRANSFORM IMAGE TO 8UNIT
            kinect_depth_processor.transform_dtype()

            # INVERT DATA TO GENERATE INVERSE COLOR MAP
            kinect_depth_processor.invert()

            # APPLY COLORMAP
            kinect_depth_processor.apply_colormap()

            projector_screen.update_window_image_calibrate(window_name="Test projected Image DEPTH",
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

    projector_application(projector_screen=projector_screen, kinect=kinect)


if __name__ == '__main__':
    main()
