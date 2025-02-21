import argparse
import logging
import os
import sys

from calibrations.CalibrateStereoFile import CalibrationStereoClass
from image_management.ImageObject import ImageObject
from image_management.ImageTransformerDepth import ImageTransformerDepth
from image_management.ImageTransformerIR import ImageTransformerIR
from image_management.ImageTransformerRGB import ImageTransformerRGB
from image_manager.ImageGenerator import ImageGenerator
from interfaces.CalibrateKinectInterface import instantiate_calibrate_kinect_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectController
from literals import IMAGE_BASE_PATH, NOT_FOUND_IMAGE_NAME, IMAGE_KINECT_SAVE_PATH, CALIBRATE_PATTERN_IMAGES, \
    KinectFrames
from utils import generate_relative_path


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
    for type_image in CALIBRATE_PATTERN_IMAGES:
        tmp_kinect_image_path = generate_relative_path([path, type_image])
        if not os.path.exists(tmp_kinect_image_path):
            logging.debug(f"Not found images in path: {tmp_kinect_image_path}")
            return previous_images

        for image_file in os.listdir(tmp_kinect_image_path):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(tmp_kinect_image_path, image_file)
                if type_image == KinectFrames.DEPTH.name:
                    image_obj = ImageObject(image_absolute_path=image_path, image_transform_class=ImageTransformerDepth)
                elif type_image == KinectFrames.INFRARED.name:
                    image_obj = ImageObject(image_absolute_path=image_path, image_transform_class=ImageTransformerIR)
                else:
                    image_obj = ImageObject(image_absolute_path=image_path, image_transform_class=ImageTransformerRGB)

                if image_file not in previous_images.keys():
                    previous_images[image_file] = {}
                previous_images[image_file][type_image] = image_obj

    return previous_images


def save_kinect_images(path, images_map):
    for image_name, image_map in images_map.items():
        for type_image in CALIBRATE_PATTERN_IMAGES:
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
    not_found_image = ImageObject(
        image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, NOT_FOUND_IMAGE_NAME])).image

    # Initiate process
    projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    app = instantiate_calibrate_kinect_interface(kinect=kinect, previous_images=previous_images,
                                                 not_found_image=not_found_image, principal_screen=principal_screen)

    # Manage Information returned
    logging.info("App finished, checking camera information")

    if save_new_images:
        save_kinect_images(path=IMAGE_KINECT_SAVE_PATH, images_map=app.photos_taken)

    for type_image in CALIBRATE_PATTERN_IMAGES:
        kinect.kinect_calibrations[type_image].set_calibrations(calibration=app.camera_information[type_image])

    # CALIBRATE STEREO
    # TODO CHECK IF THIS IS CORRECT OR NOT
    kinect.kinect_calibrations[KinectFrames.DEPTH.name].set_calibrations(
        calibration=app.camera_information[KinectFrames.INFRARED.name])
    kinect.save_calibrations()

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
        kinect = KinectController(kinect_frames=[KinectFrames.COLOR, KinectFrames.DEPTH, KinectFrames.INFRARED])
    except Exception as error:
        logging.error(f"Error trying to instantiate screens/kinect: {error}")
        raise error

    # Initiate calibrations
    try:
        calibrate_kinect(kinect=kinect, principal_screen=principal_screen,
                         projector_screen=projector_screen, use_previous_images=True,
                         save_new_images=True)

    except Exception as error:
        logging.error(f"Error in manual calibrations: {error}")
    finally:
        principal_screen.close_windows()
        projector_screen.close_windows()
        kinect.close()


if __name__ == '__main__':
    main()
