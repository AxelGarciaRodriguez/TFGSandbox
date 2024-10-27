import argparse
import logging
import sys
import threading
from copy import deepcopy
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
from image_manager.ImageProcessorDepth import ImageProcessorDepth
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectController
from literals import KinectFrames

# GLOBAL VARIABLES
min_depth = None
max_depth = None


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
    global min_depth, max_depth

    # INSTANTIATE PROJECTOR APP
    previous_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)

    kernel = np.ones((3, 3), np.uint8)

    # PROJECTOR
    projector_screen.create_window_calibrate(window_name="Projector Window", image=previous_depth, fullscreen=True)
    while projector_screen.check_if_window_active(window_name="Projector Window"):
        kinect_image_depth_no_focus = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH,
                                                                 avoid_camera_focus=True)
        if kinect_image_depth_no_focus is not None:
            kinect_depth_processor_no_focus = ImageProcessorDepth(image=kinect_image_depth_no_focus)

            # REMOVE ZEROS (INTERPOLATE)
            kinect_depth_processor_no_focus.remove_zeros()

            # COMBINE WITH PREVIOUS IMAGE (APPLY MASK)
            mask_out_of_range = ((kinect_depth_processor_no_focus.image > max_depth) | (
                        kinect_depth_processor_no_focus.image < min_depth)).astype(np.uint8)
            mask_with_neighbors = cv2.dilate(mask_out_of_range, kernel, iterations=3)

            kinect_depth_processor_no_focus.image = np.where(mask_with_neighbors == 1, previous_depth,
                                                             kinect_depth_processor_no_focus.image)

            umbral_noise = 5
            kinect_depth_processor_no_focus.image = np.where(
                (mask_with_neighbors == 0) &
                (np.abs(kinect_depth_processor_no_focus.image - previous_depth) < umbral_noise),
                previous_depth,
                kinect_depth_processor_no_focus.image
            )

            umbral_medium_noise = 15
            kinect_depth_processor_no_focus.image = np.where(
                (mask_with_neighbors == 0) &
                (umbral_noise <= np.abs(kinect_depth_processor_no_focus.image - previous_depth)) &
                (np.abs(kinect_depth_processor_no_focus.image - previous_depth) <= umbral_medium_noise),
                (previous_depth * 0.9 + kinect_depth_processor_no_focus.image * 0.1),
                kinect_depth_processor_no_focus.image
            )

            umbral_big_noise = 30
            kinect_depth_processor_no_focus.image = np.where(
                (mask_with_neighbors == 0) &
                (umbral_medium_noise <= np.abs(kinect_depth_processor_no_focus.image - previous_depth)) &
                (np.abs(kinect_depth_processor_no_focus.image - previous_depth) <= umbral_big_noise),
                (previous_depth + kinect_depth_processor_no_focus.image) // 2,
                kinect_depth_processor_no_focus.image
            )

            # SAVE PREVIOUS IMAGE
            previous_depth = deepcopy(kinect_depth_processor_no_focus.image)

            # APPLY GAUSSIAN FILTER
            kinect_depth_processor_no_focus.degaussing(ksize=(9, 9))

            # NORMALIZE IMAGE
            img_depth_normalized = ((kinect_depth_processor_no_focus.image - min_depth) / (max_depth - min_depth)) * 255.0
            kinect_depth_processor_no_focus.update(image=img_depth_normalized)

            # APPLY CAMERA FOCUS
            kinect_image_depth = kinect.apply_camera_focus(kinect_frame=KinectFrames.DEPTH,
                                                           image=kinect_depth_processor_no_focus.image)

            kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)

            # TRANSFORM IMAGE TO 8UNIT
            kinect_depth_processor.transform_dtype()

            levels = np.arange(np.min(kinect_depth_processor.image), np.max(kinect_depth_processor.image), step=10)
            smooth_contours = []
            for level in levels:
                mask_binary = (kinect_depth_processor.image == level).astype(np.uint8)
                contours_level, _ = cv2.findContours(mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours_level:
                    if len(contour) >= 20:
                        epsilon = 0.0001 * cv2.arcLength(contour, True)
                        contour_suave = cv2.approxPolyDP(contour, epsilon, True)

                        smooth_contours.append(contour_suave)

            # INVERT DATA TO GENERATE INVERSE COLOR MAP
            kinect_depth_processor.invert()

            # APPLY COLORMAP
            kinect_depth_processor.apply_colormap()

            kinect_depth_processor.draw_contours(contours=smooth_contours, color=(0, 0, 0), thickness=1)

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

    # GENERATE GLOBAL VARIABLES
    global min_depth, max_depth

    min_depth, max_depth = kinect.kinect_calibrations[KinectFrames.DEPTH.name].get_depth()

    # GENERATE THREADS
    principal_application_thread = threading.Thread(target=principal_application, args=(principal_screen,))
    projector_application_thread = threading.Thread(target=projector_application, args=(projector_screen, kinect))

    principal_application_thread.start()
    projector_application_thread.start()

    principal_application_thread.join()
    projector_application_thread.join()


if __name__ == '__main__':
    main()
