import argparse
import logging
import os
import sys

import cv2
import numpy as np

from image_manager.ImageBase import ImageBase
from image_manager.ImageGenerator import ImageGenerator
from image_manager.ImageProcessor import ImageProcessor
from interfaces.CalibrateKinectProjectorInterface import instantiate_calibrate_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectController, KinectFrames
from literals import IMAGE_BASE_PATH, PATTERN_IMAGE_NAME, NOT_FOUND_IMAGE_NAME, IMAGE_KINECT_SAVE_PATH, RGB_IMAGES_KEY, \
    DEPTH_IMAGES_KEY, IMAGE_PROJECTOR_SAVE_PATH, KINECT_CAMERA_CALIBRATION_VARIABLE, \
    KINECT_CAMERA_DISTORSION_VARIABLE, KINECT_CAMERA_ROTATION_VARIABLE, KINECT_CAMERA_TRASLATION_VARIABLE, \
    KINECT_CALIBRATION_PATH, KINECT_CALIBRATION_FILENAME, PROJECTOR_CAMERA_CALIBRATION_VARIABLE, \
    PROJECTOR_CAMERA_DISTORSION_VARIABLE, PROJECTOR_CAMERA_ROTATION_VARIABLE, PROJECTOR_CAMERA_TRASLATION_VARIABLE, \
    PROJECTOR_CALIBRATION_PATH, PROJECTOR_CALIBRATION_FILENAME
from utils import generate_relative_path, transform_cords


def get_args():
    parser = argparse.ArgumentParser(prog="calibrate_cameras",
                                     description='Initialize calibrations for kinect-projector pair',
                                     epilog='',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--logging', help='Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40 or CRITICAL=50)',
                        type=int, required=False, default=logging.INFO)

    args, unknown = parser.parse_known_args()
    return args


def load_kinect_images(path):
    previous_images = {}
    for type_image in [RGB_IMAGES_KEY, DEPTH_IMAGES_KEY]:
        tmp_kinect_image_path = generate_relative_path([path, type_image])
        if not os.path.exists(tmp_kinect_image_path):
            logging.debug(f"Not found images in path: {tmp_kinect_image_path}")
            return previous_images

        for image_file in os.listdir(tmp_kinect_image_path):
            image_processor = ImageProcessor(image_absolute_path=os.path.join(tmp_kinect_image_path, image_file))
            if image_file not in previous_images.keys():
                previous_images[image_file] = {}
            previous_images[image_file][type_image] = image_processor

    return previous_images


def save_kinect_images(path, images_map):
    for image_name, image_map in images_map.items():
        for type_image in [RGB_IMAGES_KEY, DEPTH_IMAGES_KEY]:
            tmp_kinect_image_path = generate_relative_path([path, type_image])
            if not os.path.exists(tmp_kinect_image_path):
                os.makedirs(tmp_kinect_image_path)
            if image_name not in os.listdir(tmp_kinect_image_path):
                image_map[type_image].restore()
                image_map[type_image].save(output_path=os.path.join(tmp_kinect_image_path, image_name))


def calibrate_kinect(kinect, principal_screen, projector_screen, use_previous_images, save_new_images):
    logging.info("Kinect will be calibrated using chess board pattern")

    previous_images = {}
    if use_previous_images:
        previous_images = load_kinect_images(path=IMAGE_KINECT_SAVE_PATH)

    # Read necessary images for calibration
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    not_found_image = ImageBase(
        image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, NOT_FOUND_IMAGE_NAME])).image

    # Initiate process
    projector_window = projector_screen.create_window(window_name="FullScreen Projector",
                                                      image=background_color_image, fullscreen=True)

    app = instantiate_calibrate_interface(kinect=kinect, projector_window=projector_window,
                                          projector_screen=projector_screen,
                                          previous_images=previous_images, pattern_image=None,
                                          not_found_image=not_found_image)

    # Manage Information returned
    logging.info("App finished, checking object points, image points and depth points")

    if save_new_images:
        save_kinect_images(path=IMAGE_KINECT_SAVE_PATH, images_map=app.photos_taken)

    # Calibrate camera
    ret, camera_matrix, camera_distortion, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=list(app.object_points.values()),
        imagePoints=list(app.image_points.values()),
        imageSize=app.image_shape,
        cameraMatrix=None,
        distCoeffs=None)

    arguments_saved = {
        KINECT_CAMERA_CALIBRATION_VARIABLE: camera_matrix, KINECT_CAMERA_DISTORSION_VARIABLE: camera_distortion,
        KINECT_CAMERA_ROTATION_VARIABLE: rvecs, KINECT_CAMERA_TRASLATION_VARIABLE: tvecs
    }
    kinect_calibration_path = generate_relative_path([KINECT_CALIBRATION_PATH])
    if not os.path.exists(kinect_calibration_path):
        os.makedirs(kinect_calibration_path)
    np.savez(os.path.join(kinect_calibration_path, KINECT_CALIBRATION_FILENAME), **arguments_saved)

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


def calibrate_projector(kinect, principal_screen, projector_screen, use_previous_images, save_new_images):
    logging.info("Kinect-Projector system will be calibrated using chess board pattern")

    previous_images = {}
    if use_previous_images:
        previous_images = load_kinect_images(path=IMAGE_PROJECTOR_SAVE_PATH)

    # Read necessary images for calibration
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    pattern_image = ImageBase(image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, PATTERN_IMAGE_NAME])).image
    not_found_image = ImageBase(
        image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, NOT_FOUND_IMAGE_NAME])).image

    # Initiate process
    projector_window = projector_screen.create_window(window_name="FullScreen Projector",
                                                      image=background_color_image, fullscreen=True)

    app = instantiate_calibrate_interface(kinect=kinect, projector_window=projector_window,
                                          projector_screen=projector_screen,
                                          previous_images=previous_images, pattern_image=pattern_image,
                                          not_found_image=not_found_image)

    # Manage Information returned
    logging.info("App finished, checking object points, image points and depth points")

    if save_new_images:
        save_kinect_images(path=IMAGE_PROJECTOR_SAVE_PATH, images_map=app.photos_taken)

    # Calibrate camera
    ret, camera_matrix, camera_distortion, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=list(app.object_points.values()),
        imagePoints=list(app.image_points.values()),
        imageSize=app.image_shape,
        cameraMatrix=None,
        distCoeffs=None)

    arguments_saved = {
        PROJECTOR_CAMERA_CALIBRATION_VARIABLE: camera_matrix, PROJECTOR_CAMERA_DISTORSION_VARIABLE: camera_distortion,
        PROJECTOR_CAMERA_ROTATION_VARIABLE: rvecs, PROJECTOR_CAMERA_TRASLATION_VARIABLE: tvecs
    }
    projector_calibration_path = generate_relative_path([PROJECTOR_CALIBRATION_PATH])
    if not os.path.exists(projector_calibration_path):
        os.makedirs(projector_calibration_path)
    np.savez(os.path.join(projector_calibration_path, PROJECTOR_CALIBRATION_FILENAME), **arguments_saved)

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


def calibrate_stereo(kinect, principal_screen, projector_screen, use_previous_images, save_new_images):
    # CALIBRATE CAMERA
    logging.info("Kinect will be calibrated using chess board pattern")

    previous_images = {}
    if use_previous_images:
        previous_images = load_kinect_images(path=IMAGE_KINECT_SAVE_PATH)

    # Read necessary images for calibration
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    not_found_image = ImageBase(
        image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, NOT_FOUND_IMAGE_NAME])).image

    # Initiate process
    projector_window = projector_screen.create_window(window_name="FullScreen Projector",
                                                      image=background_color_image, fullscreen=True)

    app_kinect = instantiate_calibrate_interface(kinect=kinect, projector_window=projector_window,
                                                 projector_screen=projector_screen,
                                                 previous_images=previous_images, pattern_image=None,
                                                 not_found_image=not_found_image)

    # Manage Information returned
    logging.info("App finished, checking object points, image points and depth points")

    if save_new_images:
        save_kinect_images(path=IMAGE_KINECT_SAVE_PATH, images_map=app_kinect.photos_taken)

    # Calibrate camera
    ret, camera_matrix, camera_distortion, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=list(app_kinect.object_points.values()),
        imagePoints=list(app_kinect.image_points.values()),
        imageSize=app_kinect.image_shape,
        cameraMatrix=None,
        distCoeffs=None)

    arguments_saved = {
        KINECT_CAMERA_CALIBRATION_VARIABLE: camera_matrix, KINECT_CAMERA_DISTORSION_VARIABLE: camera_distortion,
        KINECT_CAMERA_ROTATION_VARIABLE: rvecs, KINECT_CAMERA_TRASLATION_VARIABLE: tvecs
    }
    kinect_calibration_path = generate_relative_path([KINECT_CALIBRATION_PATH])
    if not os.path.exists(kinect_calibration_path):
        os.makedirs(kinect_calibration_path)
    np.savez(os.path.join(kinect_calibration_path, KINECT_CALIBRATION_FILENAME), **arguments_saved)

    # CALIBRATE PROJECTOR
    previous_images = {}
    if use_previous_images:
        previous_images = load_kinect_images(path=IMAGE_PROJECTOR_SAVE_PATH)

    pattern_image = ImageBase(image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, PATTERN_IMAGE_NAME])).image
    app_projector = instantiate_calibrate_interface(kinect=kinect, projector_window=projector_window,
                                                    projector_screen=projector_screen,
                                                    previous_images=previous_images, pattern_image=pattern_image,
                                                    not_found_image=not_found_image)

    if save_new_images:
        save_kinect_images(path=IMAGE_PROJECTOR_SAVE_PATH, images_map=app_projector.photos_taken)

    # Calibrate camera
    ret, projector_matrix, projector_distortion, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=list(app_projector.object_points.values()),
        imagePoints=list(app_projector.image_points.values()),
        imageSize=app_projector.image_shape,
        cameraMatrix=None,
        distCoeffs=None)

    arguments_saved = {
        PROJECTOR_CAMERA_CALIBRATION_VARIABLE: camera_matrix, PROJECTOR_CAMERA_DISTORSION_VARIABLE: camera_distortion,
        PROJECTOR_CAMERA_ROTATION_VARIABLE: rvecs, PROJECTOR_CAMERA_TRASLATION_VARIABLE: tvecs
    }
    projector_calibration_path = generate_relative_path([PROJECTOR_CALIBRATION_PATH])
    if not os.path.exists(projector_calibration_path):
        os.makedirs(projector_calibration_path)
    np.savez(os.path.join(projector_calibration_path, PROJECTOR_CALIBRATION_FILENAME), **arguments_saved)

    # CALIBRATE STEREO
    logging.info("Read calibration information")

    ret, camera_matrix, dist_coefs, projector_matrix, projector_dist_coefs, R, T, E, F = cv2.stereoCalibrate(
        list(app_kinect.object_points.values()), list(app_kinect.image_points.values()),
        list(app_projector.image_points.values()), camera_matrix, camera_distortion, projector_matrix,
        projector_distortion, app_projector.image_shape
    )

    print("Projector matrix:\n", projector_matrix)
    print("Distortion coefficients:\n", projector_dist_coefs)
    print("Rotation matrix:\n", R)
    print("Translation vector:\n", T)

    ######################### TEST POINT PROJECTION

    kinect_rgb_image = kinect.get_image(kinect_frame=KinectFrames.COLOR)
    kinect_depth_image = kinect.get_image(kinect_frame=KinectFrames.DEPTH)

    x, y = (
        kinect_rgb_image.shape[1] // 2,
        kinect_rgb_image.shape[0] // 2
    )

    x_transformed, y_transformed = transform_cords(point=(x, y), original_size=kinect_rgb_image.shape[:2],
                                                   new_size=kinect_depth_image.shape[:2])
    z = kinect_depth_image[y_transformed, x_transformed]

    point_3d_kinect = np.array([x, y, z], dtype=np.float32).reshape(3, 1)
    point_3d_projector = np.dot(R, point_3d_kinect) + T

    point_2d_projector, _ = cv2.projectPoints(point_3d_projector.T.reshape(-1, 1, 3), np.zeros((3, 1)),
                                              np.zeros((3, 1)), projector_matrix, projector_dist_coefs)

    x_projected = point_2d_projector[0][0][0]
    y_projected = point_2d_projector[0][0][1]

    # Generate images with points
    base_image = ImageProcessor(image=ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution))
    base_image.draw_point(point=(x_projected, y_projected))
    projector_window.update_image(image=base_image.image)

    #########################

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


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
        kinect = KinectController(kinect_frames=[KinectFrames.COLOR, KinectFrames.DEPTH])
    except Exception as error:
        logging.error(f"Error trying to instantiate screens/kinect: {error}")
        raise error

    # Initiate calibrations
    try:
        calibrate_stereo(kinect=kinect, principal_screen=principal_screen,
                         projector_screen=projector_screen, use_previous_images=True,
                         save_new_images=True)

        # calibrate_kinect(kinect=kinect, principal_screen=principal_screen,
        #                  projector_screen=projector_screen, use_previous_images=True,
        #                  save_new_images=True)
        #
        # calibrate_projector(kinect=kinect, principal_screen=principal_screen,
        #                     projector_screen=projector_screen, use_previous_images=True,
        #                     save_new_images=True)

        # calibrate_stereo()
    except Exception as error:
        logging.error(f"Error in manual calibrations: {error}")
    finally:
        principal_screen.close_windows()
        projector_screen.close_windows()
        kinect.close()


if __name__ == '__main__':
    main()
