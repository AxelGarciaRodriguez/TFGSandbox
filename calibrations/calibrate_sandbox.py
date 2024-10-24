import argparse
import logging
import sys
import time

from image_manager.ImageGenerator import ImageGenerator
from image_manager.ImageProcessorDepth import ImageProcessorDepth
from image_manager.ImageProcessorIR import ImageProcessorIR
from image_manager.ImageProcessorRGB import ImageProcessorRGB
from interfaces.DrawPolygonInterface import instantiate_draw_polygon_interface
from interfaces.MoveProjectorPointsInterface import instantiate_move_projector_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectFrames, KinectController
from literals import BOX_HEIGHT
from utils import generate_cords


def get_args():
    parser = argparse.ArgumentParser(prog="calibrate_sandbox",
                                     description='Initialize calibrations for sandbox detection',
                                     epilog='',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--logging', help='Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40 or CRITICAL=50)',
                        type=int, required=False, default=logging.INFO)

    args, unknown = parser.parse_known_args()
    return args


def calibrate_kinect_sandbox_rgb(kinect, principal_screen, projector_screen):
    kinect_frame = KinectFrames.COLOR
    logging.info(f"Calibrate {kinect_frame.name} image")
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    kinect_image = kinect.get_image_calibrate(kinect_frame=kinect_frame, avoid_camera_focus=True)
    kinect_processor = ImageProcessorRGB(image=kinect_image)
    kinect_processor.invert()

    app_draw_polygon = instantiate_draw_polygon_interface(image=kinect_processor.image,
                                                          depth_image=None,
                                                          principal_screen=principal_screen)

    kinect.kinect_calibrations[kinect_frame.name].calculate_homography(
        cords=app_draw_polygon.points,
        original_cords=generate_cords(size=(kinect_processor.height, kinect_processor.width)))

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


def calibrate_kinect_sandbox_ir_depth(kinect, principal_screen, projector_screen):
    kinect_frame = KinectFrames.INFRARED
    logging.info(f"Calibrate {kinect_frame.name} image")
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)
    kinect_processor_depth = ImageProcessorDepth(image=kinect_image_depth)
    kinect_processor_depth.remove_zeros()

    kinect_image_ir = kinect.get_image_calibrate(kinect_frame=kinect_frame, avoid_camera_focus=True)
    kinect_processor_ir = ImageProcessorIR(image=kinect_image_ir)
    kinect_processor_ir.normalize()
    kinect_processor_ir.transform_dtype()

    app_draw_polygon = instantiate_draw_polygon_interface(image=kinect_processor_ir.image,
                                                          depth_image=kinect_processor_depth.image,
                                                          principal_screen=principal_screen)

    points_2d = []
    min_depth = None
    for point_3d in app_draw_polygon.points:
        x, y, z = point_3d
        points_2d.append((x, y))

        if not min_depth or min_depth < z:
            min_depth = z

    # INFRARED CALIBRATION
    kinect.kinect_calibrations[kinect_frame.name].calculate_homography(
        cords=points_2d,
        original_cords=generate_cords(size=(kinect_processor_ir.height, kinect_processor_ir.width)))

    # DEPTH CALIBRATION
    kinect_frame = KinectFrames.DEPTH
    kinect.kinect_calibrations[kinect_frame.name].calculate_homography(
        cords=points_2d,
        original_cords=generate_cords(size=(kinect_processor_depth.height, kinect_processor_depth.width)))

    kinect.kinect_calibrations[kinect_frame.name].min_depth = int(min_depth)
    # GENERATE MAX DEPTH BY APPROXIMATE VALUE
    kinect.kinect_calibrations[kinect_frame.name].max_depth = int(min_depth + BOX_HEIGHT)

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


def calibrate_projector_sandbox(kinect, principal_screen, projector_screen):
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_window = projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image,
                                                      fullscreen=True)

    base_kinect_points = kinect.kinect_calibrations[KinectFrames.COLOR.name].cords
    app_move_projector = instantiate_move_projector_interface(kinect=kinect, rgb_points=base_kinect_points,
                                                              projector_window=projector_window,
                                                              principal_screen=principal_screen)

    projector_screen.calibration.calculate_homography(
        cords=app_move_projector.points,
        original_cords=generate_cords(size=(projector_screen.height_resolution, projector_screen.width_resolution)))

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


def save_calibrations(kinect, projector_screen):
    logging.info("Saving calibration files")
    kinect.save_calibrations()
    projector_screen.save_calibration()


def test_calibrations(kinect, principal_screen, projector_screen):
    logging.info("Testing calibrations")

    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    # KINECT TEST
    kinect_image_color = kinect.get_image_calibrate(kinect_frame=KinectFrames.COLOR)
    kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH)
    kinect_image_infrared = kinect.get_image_calibrate(kinect_frame=KinectFrames.INFRARED)

    kinect_color_processor = ImageProcessorRGB(image=kinect_image_color)
    kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)
    kinect_infrared_processor = ImageProcessorIR(image=kinect_image_infrared)

    # INFRARED MANAGEMENT
    kinect_infrared_processor.normalize()
    kinect_infrared_processor.transform_dtype()

    # DEPTH MANAGEMENT
    min_depth, max_depth = kinect.kinect_calibrations[KinectFrames.DEPTH.name].get_depth()

    kinect_depth_processor.remove_zeros()
    kinect_depth_processor.degaussing()
    kinect_depth_processor.remove_data_between_distance_option_b(min_depth=min_depth, max_depth=max_depth)
    img_depth_normalized = ((kinect_depth_processor.image - min_depth) / (max_depth - min_depth)) * 255.0
    kinect_depth_processor.update(image=img_depth_normalized)
    kinect_depth_processor.transform_dtype()
    kinect_depth_processor.invert()
    kinect_depth_processor.apply_colormap()

    principal_screen.create_window(window_name="Color Focused", image=kinect_color_processor.image)
    principal_screen.create_window(window_name="Depth Focused", image=kinect_depth_processor.image)
    principal_screen.create_window(window_name="Infrared Focused", image=kinect_infrared_processor.image)

    while (principal_screen.check_if_window_active(window_name="Infrared Focused") or
           principal_screen.check_if_window_active(window_name="Depth Focused") or
           principal_screen.check_if_window_active(window_name="Color Focused")
    ):
        time.sleep(1)

    # PROJECTOR TEST
    projector_screen.create_window_calibrate(window_name="Test projected Image DEPTH",
                                             image=kinect_depth_processor.image, fullscreen=True)
    while projector_screen.check_if_window_active(window_name="Test projected Image DEPTH"):
        kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH)
        if kinect_image_depth is not None:
            kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)
            kinect_depth_processor.remove_zeros()
            kinect_depth_processor.degaussing()
            kinect_depth_processor.remove_data_between_distance_option_b(min_depth=min_depth, max_depth=max_depth)
            img_depth_normalized = ((kinect_depth_processor.image - min_depth) / (max_depth - min_depth)) * 255.0
            kinect_depth_processor.update(image=img_depth_normalized)
            kinect_depth_processor.transform_dtype()
            kinect_depth_processor.invert()
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
        kinect = KinectController(kinect_frames=[KinectFrames.COLOR, KinectFrames.DEPTH, KinectFrames.INFRARED])
    except Exception as error:
        logging.error(f"Error trying to instantiate screens/kinect: {error}")
        raise error

    # Initiate calibrations
    try:
        calibrate_kinect_sandbox_rgb(kinect=kinect, principal_screen=principal_screen,
                                     projector_screen=projector_screen)
        calibrate_kinect_sandbox_ir_depth(kinect=kinect, principal_screen=principal_screen,
                                          projector_screen=projector_screen)
        calibrate_projector_sandbox(kinect=kinect, principal_screen=principal_screen, projector_screen=projector_screen)
        save_calibrations(kinect=kinect, projector_screen=projector_screen)
        test_calibrations(kinect=kinect, principal_screen=principal_screen, projector_screen=projector_screen)

    except Exception as error:
        logging.error(f"Error in manual calibrations: {error}")
    finally:
        principal_screen.close_windows()
        projector_screen.close_windows()
        kinect.close()


if __name__ == '__main__':
    main()
