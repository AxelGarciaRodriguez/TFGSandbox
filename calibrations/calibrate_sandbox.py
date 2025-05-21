import argparse
import ctypes
import logging
import sys
import time

import numpy as np

from image_management.ImageObject import ImageObject
from image_management.ImageTransformerDepth import ImageTransformerDepth
from image_management.ImageTransformerIR import ImageTransformerIR
from image_management.ImageTransformerRGB import ImageTransformerRGB
from image_management.ImageGenerator import ImageGenerator
from interfaces.DrawPolygonInterface import instantiate_draw_polygon_interface
from interfaces.MoveProjectorPointsInterface import instantiate_move_projector_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectFrames, KinectController
from kinect_module.PyKinectV2 import _DepthSpacePoint
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
    inverted_image = ImageTransformerRGB.invert(image=kinect_image)
    kinect_processor = ImageObject(image=inverted_image)

    app_draw_polygon = instantiate_draw_polygon_interface(image=kinect_processor.image,
                                                          depth_image=None,
                                                          principal_screen=principal_screen)

    kinect.kinect_calibrations[kinect_frame.name].calculate_homography(
        cords=app_draw_polygon.points,
        original_cords=generate_cords(size=(kinect_processor.height, kinect_processor.width)))

    # Close all windows in main screen and projector screen
    principal_screen.close_windows()
    projector_screen.close_windows()


def depth_image_to_depth_frame(depth_image):
    depth_image = depth_image.astype(np.uint16)

    depth_array = depth_image.flatten()
    depth_array_ptr = (ctypes.c_uint16 * len(depth_array))(*depth_array)

    return depth_array_ptr


def calibrate_kinect_sandbox_ir_depth(kinect, principal_screen, projector_screen):
    # kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True, avoid_camera_matrix=True)
    # kinect_processor_depth = ImageObject(image=kinect_image_depth)
    #
    # kinect_image_depth_zeros_removed = ImageTransformerDepth.remove_zeros(image=kinect_image_depth)
    # kinect_processor_depth.update(image=kinect_image_depth_zeros_removed)
    #
    # color2depth_points_type = _DepthSpacePoint * int(1920 * 1080)
    # color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(_DepthSpacePoint))
    #
    # kinect.kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424),
    #                                          depth_image_to_depth_frame(depth_image=kinect_processor_depth.image),
    #                                          ctypes.c_uint(1920 * 1080),
    #                                          color2depth_points)
    #
    # min_depth = None
    # points_2d = []
    # for color_point in kinect.kinect_calibrations[KinectFrames.COLOR.name].cords:
    #     color_coords = color2depth_points[int(color_point[1]) * 1920 + int(color_point[0])]
    #     depth_x = int(color_coords.x)
    #     depth_y = int(color_coords.y)
    #     points_2d.append((depth_x, depth_y))
    #
    #     depth_z = kinect_processor_depth.image[int(depth_y), int(depth_x)]
    #     if not min_depth or min_depth < depth_z:
    #         min_depth = depth_z

    kinect_frame = KinectFrames.INFRARED
    logging.info(f"Calibrate {kinect_frame.name} image")
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    kinect_image_depth = kinect.get_image_calibrate(kinect_frame=KinectFrames.DEPTH, avoid_camera_focus=True)
    kinect_image_depth = ImageTransformerDepth.remove_zeros(image=kinect_image_depth)
    kinect_processor_depth = ImageObject(image=kinect_image_depth)

    kinect_image_ir = kinect.get_image_calibrate(kinect_frame=kinect_frame, avoid_camera_focus=True)
    kinect_image_ir = ImageTransformerIR.normalize(image=kinect_image_ir)
    kinect_image_ir = ImageTransformerIR.transform_dtype(image=kinect_image_ir)
    kinect_processor_ir = ImageObject(image=kinect_image_ir)

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

    kinect_color_processor = ImageObject(image=kinect_image_color)
    kinect_depth_processor = ImageObject(image=kinect_image_depth)
    kinect_infrared_processor = ImageObject(image=kinect_image_infrared)

    # INFRARED MANAGEMENT
    image = ImageTransformerIR.normalize(image=kinect_infrared_processor.image)
    image = ImageTransformerIR.transform_dtype(image=image)
    kinect_infrared_processor.update(image=image)

    # DEPTH MANAGEMENT
    min_depth, max_depth = kinect.kinect_calibrations[KinectFrames.DEPTH.name].get_depth()

    image = ImageTransformerDepth.remove_zeros(image=kinect_depth_processor.image)
    image = ImageTransformerDepth.degaussing(image=image)
    image = ImageTransformerDepth.remove_data_between_distance(image=image, min_depth=min_depth, max_depth=max_depth)
    image = ImageTransformerDepth.normalize_between_distance(image=image, min_depth=min_depth, max_depth=max_depth)
    image = ImageTransformerDepth.transform_dtype(image=image)
    image = ImageTransformerDepth.invert(image=image)
    image = ImageTransformerDepth.apply_colormap(image=image)

    kinect_depth_processor.update(image=image)

    principal_screen.create_window(window_name="Color Focused", image=kinect_color_processor.image)
    principal_screen.create_window(window_name="Depth Focused", image=kinect_depth_processor.image)
    principal_screen.create_window(window_name="Infrared Focused", image=kinect_infrared_processor.image)

    while (principal_screen.check_if_window_active(window_name="Infrared Focused") or
           principal_screen.check_if_window_active(window_name="Depth Focused") or
           principal_screen.check_if_window_active(window_name="Color Focused")
    ):
        time.sleep(1)


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
        # calibrate_projector_sandbox(kinect=kinect, principal_screen=principal_screen, projector_screen=projector_screen)
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
