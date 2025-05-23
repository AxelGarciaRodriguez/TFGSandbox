import argparse
import logging
import sys
import threading

import cv2
import numpy as np

from image_management.ApplicationController import SharedConfig
from image_management.ImageTransformerDepth import ImageTransformerDepth
from interfaces.PrincipalApplicationInterface import instantiate_principal_application_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectController
from literals import KinectFrames, ConfigControllerEnum, CONTOURS_EPSILON_FACTOR, CONTOURS_MIN_AREA


def get_args():
    parser = argparse.ArgumentParser(prog="app",
                                     description='Principal application',
                                     epilog='',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--logging', help='Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40 or CRITICAL=50)',
                        type=int, required=False, default=logging.INFO)

    args, unknown = parser.parse_known_args()
    return args


def calculate_smoothed_contours(image, config_values):
    smooth_contours = []
    for mask in ImageTransformerDepth.get_masks_by_steps(image=image, step_value=config_values[
        ConfigControllerEnum.CONTOURS_LEVEL_STEPS.name]):
        contours = ImageTransformerDepth.find_contours(image=mask, mode=cv2.RETR_TREE, flags=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = ImageTransformerDepth.get_contour_area(contour=contour)
            if area >= CONTOURS_MIN_AREA:
                smooth_contours.append(ImageTransformerDepth.approx_poly(points=contour,
                                                                         epsilon_factor=CONTOURS_EPSILON_FACTOR))

    return smooth_contours


def projector_application(projector_screen, kinect, config: SharedConfig):
    previous_min_depth = config.get_value(ConfigControllerEnum.MIN_DEPTH.name)
    previous_max_depth = config.get_value(ConfigControllerEnum.MAX_DEPTH.name)

    # GET FIRST DEPHT IMAGE FOR COMBINED IMAGES IN PROCESS
    depth_image = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)
    depth_image_without_zeros = ImageTransformerDepth.remove_zeros(image=depth_image)
    depth_image_set_distance_datas = ImageTransformerDepth.set_data_between_distance(image=depth_image_without_zeros,
                                                                                     min_depth=previous_min_depth,
                                                                                     max_depth=previous_max_depth)

    previous_depth = depth_image_set_distance_datas

    # CREATE WINDOW SCREEN
    projector_screen.create_window_calibrate(window_name="Projector Window", image=previous_depth, fullscreen=True)
    while projector_screen.check_if_window_active(window_name="Projector Window"):
        depth_image = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)
        if depth_image is not None:
            config_values = config.get_values()

            if (config_values[ConfigControllerEnum.RESET_IMAGE.name] or
                    (previous_min_depth != config_values[ConfigControllerEnum.MIN_DEPTH.name] or
                     previous_max_depth != config_values[ConfigControllerEnum.MAX_DEPTH.name])):
                depth_image_set_distance_datas = ImageTransformerDepth.set_data_between_distance(
                    image=depth_image_without_zeros,
                    min_depth=config_values[ConfigControllerEnum.MIN_DEPTH.name],
                    max_depth=config_values[ConfigControllerEnum.MAX_DEPTH.name])

                previous_depth = depth_image_set_distance_datas
                previous_min_depth = config_values[ConfigControllerEnum.MIN_DEPTH.name]
                previous_max_depth = config_values[ConfigControllerEnum.MAX_DEPTH.name]
                config.set_value(key=ConfigControllerEnum.RESET_IMAGE.name, value=False)

            # REMOVE ERRORS IN IMAGE
            depth_image_no_zeros = ImageTransformerDepth.remove_zeros(image=depth_image)

            # COMBINE WITH PREVIOUS IMAGE DATA OUT OF THE SANDBOX RANGE (MIN AND MAX DEPTH)
            last_depth_image, mask_with_neighbors = ImageTransformerDepth.remove_data_between_distance_neighbors(
                image=depth_image_no_zeros,
                min_depth=config_values[ConfigControllerEnum.MIN_DEPTH.name],
                max_depth=config_values[ConfigControllerEnum.MAX_DEPTH.name],
                other_image=previous_depth, iterations=30)

            # REMOVE NOISE (CHANGES < 5 MM)
            condition = (mask_with_neighbors == 0) & (np.abs(last_depth_image - previous_depth) < config_values[
                ConfigControllerEnum.ERRORS_UMBRAL.name])
            value = previous_depth
            depth_image_no_errors = ImageTransformerDepth.apply_mask(image=last_depth_image, condition=condition,
                                                                     value=value)

            # COMBINE MEDIUM NOISE (CHANGES BETWEEN 5MM AND 15MM) --> 10% ACTUAL, 90% PREVIOUS
            condition = (
                    (mask_with_neighbors == 0) &
                    (config_values[ConfigControllerEnum.ERRORS_UMBRAL.name] <= np.abs(
                        depth_image_no_errors - previous_depth)) &
                    (np.abs(depth_image_no_errors - previous_depth) <= config_values[
                        ConfigControllerEnum.MEDIUM_NOISE.name])
            )
            value = previous_depth * 0.9 + depth_image_no_errors * 0.1
            depth_image_no_noise = ImageTransformerDepth.apply_mask(image=depth_image_no_errors,
                                                                    condition=condition,
                                                                    value=value)

            # REMOVE BIG NOISE (CHANGES BETWEEN 15MM AND 30MM) --> 50% ACTUAL, 50% PREVIOUS
            condition = (
                    (mask_with_neighbors == 0) &
                    (config_values[ConfigControllerEnum.MEDIUM_NOISE.name] <= np.abs(
                        depth_image_no_noise - previous_depth)) &
                    (np.abs(depth_image_no_noise - previous_depth) <= config_values[
                        ConfigControllerEnum.BIG_NOISE.name])
            )
            value = previous_depth * 0.5 + depth_image_no_noise * 0.5
            last_depth_image = ImageTransformerDepth.apply_mask(image=depth_image_no_noise, condition=condition,
                                                                value=value)

            # SAVE PREVIOUS IMAGE
            previous_depth = ImageTransformerDepth.duplicate(image=last_depth_image)

            # APPLY CAMERA FOCUS
            depth_image_transformed = kinect.apply_camera_focus(kinect_frame=KinectFrames.DEPTH,
                                                                image=last_depth_image)

            # NORMALIZE IMAGE
            depth_image_normalized = ImageTransformerDepth.normalize_between_distance(image=depth_image_transformed,
                                                                                      min_depth=config_values[
                                                                                          ConfigControllerEnum.MIN_DEPTH.name],
                                                                                      max_depth=config_values[
                                                                                          ConfigControllerEnum.MAX_DEPTH.name])

            # TRANSFORM IMAGE TO UINT8
            depth_image_uint8 = ImageTransformerDepth.transform_dtype(image=depth_image_normalized, dtype=np.uint8)

            # BLURRED IMAGE FOR CALCULATIONS
            depth_image_blurred = ImageTransformerDepth.degaussing(image=depth_image_uint8, ksize=(11, 11), sigma_x=0)

            # CONTOURS (LEVEL LINES)
            smoothed_contours = calculate_smoothed_contours(image=depth_image_blurred, config_values=config_values)

            # GENERATE COLOR IMAGE (INVERT + APPLY COLORMAP)
            depth_image_uint8_inverted = ImageTransformerDepth.invert(image=depth_image_uint8)
            colormap_image = ImageTransformerDepth.apply_colormap(image=depth_image_uint8_inverted,
                                                                  colormap=config_values[
                                                                      ConfigControllerEnum.COLORMAP.name])

            # DRAW INFORMATION IN COLORMAP IMAGE
            colormap_with_contours = ImageTransformerDepth.draw_contours(image=colormap_image, thickness=1,
                                                                         contours=smoothed_contours, color=(0, 0, 0))

            # UPDATE IMAGE PROJECTED
            final_image = colormap_with_contours
            projector_screen.update_window_image_calibrate(window_name="Projector Window", image=final_image)

            rgb_image = kinect.get_image_calibrate(kinect_frame=KinectFrames.COLOR)
            with config.lock:
                config.current_image = final_image
                if rgb_image is not None:
                    config.second_image = rgb_image


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
        kinect = KinectController(kinect_frames=[KinectFrames.DEPTH, KinectFrames.COLOR])
    except Exception as error:
        logging.error(f"Error trying to instantiate screens/kinect: {error}")
        raise error

    # GENERATE SHARED CONFIG
    config = SharedConfig()
    min_depth, max_depth = kinect.kinect_calibrations[KinectFrames.DEPTH.name].get_depth()
    config.set_value(key=ConfigControllerEnum.MIN_DEPTH.name, value=min_depth)
    config.set_value(key=ConfigControllerEnum.MAX_DEPTH.name, value=max_depth)

    projector_application_thread = threading.Thread(target=projector_application,
                                                    args=(projector_screen, kinect, config))
    projector_application_thread.start()

    instantiate_principal_application_interface(config=config, principal_screen=principal_screen)

    projector_application_thread.join()


if __name__ == '__main__':
    main()
