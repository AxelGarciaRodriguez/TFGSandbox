import argparse
import logging
import sys
import threading

import cv2
import numpy as np

from image_management.ImageTransformerDepth import ImageTransformerDepth
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectController
from literals import KinectFrames, CONTOURS_LEVEL_STEPS, CONTOURS_MIN_AREA, CONTOURS_EPSILON_FACTOR, ERRORS_UMBRAL, \
    MEDIUM_NOISE, BIG_NOISE

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


def calculate_smoothed_contours(image):
    smooth_contours = []
    for mask in ImageTransformerDepth.get_masks_by_steps(image=image, step_value=CONTOURS_LEVEL_STEPS):
        contours = ImageTransformerDepth.find_contours(image=mask, mode=cv2.RETR_TREE, flags=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = ImageTransformerDepth.get_contour_area(contour=contour)
            if area >= CONTOURS_MIN_AREA:
                smooth_contours.append(ImageTransformerDepth.approx_poly(points=contour,
                                                                         epsilon_factor=CONTOURS_EPSILON_FACTOR))

    return smooth_contours


def principal_application(principal_screen):
    pass


def projector_application(projector_screen, kinect):
    global min_depth, max_depth

    # GET FIRST DEPHT IMAGE FOR COMBINED IMAGES IN PROCESS
    depth_image = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)
    depth_image_without_zeros = ImageTransformerDepth.remove_zeros(image=depth_image)
    depth_image_set_distance_datas = ImageTransformerDepth.set_data_between_distance(image=depth_image_without_zeros,
                                                                                     min_depth=min_depth,
                                                                                     max_depth=max_depth)

    previous_depth = depth_image_set_distance_datas

    # CREATE WINDOW SCREEN
    projector_screen.create_window_calibrate(window_name="Projector Window", image=depth_image, fullscreen=True)
    while projector_screen.check_if_window_active(window_name="Projector Window"):
        depth_image = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)
        if depth_image is not None:
            # REMOVE ERRORS IN IMAGE
            depth_image_no_zeros = ImageTransformerDepth.remove_zeros(image=depth_image)

            # COMBINE WITH PREVIOUS IMAGE DATA OUT OF THE SANDBOX RANGE (MIN AND MAX DEPTH)
            depth_image_combined, mask_with_neighbors = ImageTransformerDepth.remove_data_between_distance_neighbors(
                image=depth_image_no_zeros,
                min_depth=min_depth, max_depth=max_depth,
                other_image=previous_depth, iterations=30)

            # REMOVE NOISE (CHANGES < 5 MM)
            condition = (mask_with_neighbors == 0) & (np.abs(depth_image_combined - previous_depth) < ERRORS_UMBRAL)
            value = previous_depth
            depth_image_no_errors = ImageTransformerDepth.apply_mask(image=depth_image_combined, condition=condition,
                                                                     value=value)

            # COMBINE MEDIUM NOISE (CHANGES BETWEEN 5MM AND 15MM) --> 10% ACTUAL, 90% PREVIOUS
            condition = (
                    (mask_with_neighbors == 0) &
                    (ERRORS_UMBRAL <= np.abs(depth_image_no_errors - previous_depth)) &
                    (np.abs(depth_image_no_errors - previous_depth) <= MEDIUM_NOISE)
            )
            value = previous_depth * 0.9 + depth_image_no_errors * 0.1
            depth_image_no_noise = ImageTransformerDepth.apply_mask(image=depth_image_no_errors, condition=condition,
                                                                    value=value)

            # REMOVE BIG NOISE (CHANGES BETWEEN 15MM AND 30MM) --> 50% ACTUAL, 50% PREVIOUS
            condition = (
                    (mask_with_neighbors == 0) &
                    (MEDIUM_NOISE <= np.abs(depth_image_no_noise - previous_depth)) &
                    (np.abs(depth_image_no_noise - previous_depth) <= BIG_NOISE)
            )
            value = previous_depth * 0.5 + depth_image_no_noise * 0.5
            depth_image_no_big_noise = ImageTransformerDepth.apply_mask(image=depth_image_no_noise, condition=condition,
                                                                        value=value)

            # SAVE PREVIOUS IMAGE
            previous_depth = ImageTransformerDepth.duplicate(image=depth_image_no_big_noise)

            # APPLY CAMERA FOCUS
            depth_image_transformed = kinect.apply_camera_focus(kinect_frame=KinectFrames.DEPTH,
                                                                image=depth_image_no_big_noise)

            # NORMALIZE IMAGE
            depth_image_normalized = ImageTransformerDepth.normalize_between_distance(image=depth_image_transformed,
                                                                                      min_depth=min_depth,
                                                                                      max_depth=max_depth)

            # TRANSFORM IMAGE TO UINT8
            depth_image_uint8 = ImageTransformerDepth.transform_dtype(image=depth_image_normalized, dtype=np.uint8)

            # BLURRED IMAGE FOR CALCULATIONS
            depth_image_blurred = ImageTransformerDepth.degaussing(image=depth_image_uint8, ksize=(11, 11), sigma_x=0)

            # CONTOURS (LEVEL LINES)
            smoothed_contours = calculate_smoothed_contours(image=depth_image_blurred)

            # GENERATE COLOR IMAGE (INVERT + APPLY COLORMAP)
            depth_image_uint8_inverted = ImageTransformerDepth.invert(image=depth_image_uint8)
            colormap_image = ImageTransformerDepth.apply_colormap(image=depth_image_uint8_inverted,
                                                                  colormap=cv2.COLORMAP_JET)

            # DRAW INFORMATION IN COLORMAP IMAGE
            colormap_with_contours = ImageTransformerDepth.draw_contours(image=colormap_image, thickness=1,
                                                                         contours=smoothed_contours, color=(0, 0, 0))

            # UPDATE IMAGE PROJECTED
            final_image = colormap_with_contours
            projector_screen.update_window_image_calibrate(window_name="Projector Window", image=final_image)


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
