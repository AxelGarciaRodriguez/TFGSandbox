import argparse
import logging
import sys
import time

import cv2
import numpy as np
from scipy.interpolate import griddata

from image_manager.ImageGenerator import ImageGenerator
from image_manager.ImageProcessorDepth import ImageProcessorDepth
from image_manager.ImageProcessorIR import ImageProcessorIR
from image_manager.ImageProcessorRGB import ImageProcessorRGB
from interfaces.DrawPolygonInterface import instantiate_draw_polygon_interface
from interfaces.MoveProjectorPointsInterface import instantiate_move_projector_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectFrames, KinectController
from utils import generate_cords, ordering_points, transform_cords


def get_args():
    parser = argparse.ArgumentParser(prog="calibrate_sandbox",
                                     description='Initialize calibrations for sandbox detection',
                                     epilog='',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--logging', help='Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40 or CRITICAL=50)',
                        type=int, required=False, default=logging.INFO)

    args, unknown = parser.parse_known_args()
    return args


def calibrate_kinect_sandbox(kinect, kinect_frame, principal_screen, projector_screen):
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    kinect_image = kinect.get_image(kinect_frame=kinect_frame)
    if kinect_frame == KinectFrames.COLOR:
        kinect_processor = ImageProcessorRGB(image=kinect_image)
    else:
        kinect_processor = ImageProcessorIR(image=kinect_image)
        kinect_processor.normalize()
        kinect_processor.transform_dtype()

    logging.info(f"Calibrate {kinect_frame.name} image")
    app_draw_polygon = instantiate_draw_polygon_interface(image=kinect_processor.image,
                                                          principal_screen=principal_screen)

    kinect.kinect_calibrations[kinect_frame.name].calculate_homography(
        cords=app_draw_polygon.points,
        original_cords=generate_cords(size=(kinect_processor.height, kinect_processor.width)))

    if kinect_frame != KinectFrames.COLOR:
        # Generate DEPTH calibration as IR due to cannot calibrate depth images
        kinect.kinect_calibrations[KinectFrames.DEPTH.name].calculate_homography(
            cords=app_draw_polygon.points,
            original_cords=generate_cords(size=(kinect_processor.height, kinect_processor.width)))

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

    # projector_screen.calibration.calculate_homography(
    #     cords=generate_cords(size=(projector_screen.height_resolution, projector_screen.width_resolution)),
    #     original_cords=app_move_projector.points)

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

    kinect_depth_processor.normalize()
    kinect_depth_processor.transform_dtype()
    kinect_depth_processor.degaussing()
    kinect_depth_processor.invert()
    kinect_depth_processor.apply_colormap()

    kinect_infrared_processor.normalize()
    kinect_infrared_processor.transform_dtype()

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
            kinect_depth_processor.normalize()
            kinect_depth_processor.transform_dtype()
            kinect_depth_processor.degaussing()
            kinect_depth_processor.invert()
            kinect_depth_processor.apply_colormap()

            projector_screen.update_window_image_calibrate(window_name="Test projected Image DEPTH",
                                                           image=kinect_depth_processor.image)
        time.sleep(1)


def calibrate_sandbox_old(kinect, principal_screen, projector_screen):
    # FIRST, SELECT SAND BOX IN KINECT IMAGE
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_window = projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image,
                                                      fullscreen=True)

    kinect_image_color = kinect.get_image(kinect_frame=KinectFrames.COLOR)
    kinect_image_depth = kinect.get_image(kinect_frame=KinectFrames.DEPTH)
    kinect_image_infrared = kinect.get_image(kinect_frame=KinectFrames.INFRARED)

    kinect_color_processor = ImageProcessorRGB(image=kinect_image_color)
    kinect_depth_processor = ImageProcessorDepth(image=kinect_image_depth)
    kinect_infrared_processor = ImageProcessorIR(image=kinect_image_infrared)

    kinect_infrared_processor.normalize()
    kinect_infrared_processor.transform_dtype()

    logging.info("Calibrate IR image")
    app_draw_polygon_ir = instantiate_draw_polygon_interface(image=kinect_infrared_processor.image,
                                                             principal_screen=principal_screen)

    detected_cords = np.array(app_draw_polygon_ir.points, dtype=np.float32)
    kinect_cords = np.array(
        generate_cords(size=(kinect_infrared_processor.height, kinect_infrared_processor.width)),
        dtype=np.float32)

    detected_cords_ordered = np.float32(ordering_points(kinect_cords, detected_cords))
    focus_matrix_ir = cv2.getPerspectiveTransform(detected_cords_ordered, kinect_cords)

    logging.info("Calibrate RGB image")
    app_draw_polygon_rgb = instantiate_draw_polygon_interface(image=kinect_color_processor.image,
                                                              principal_screen=principal_screen)

    detected_cords = np.array(app_draw_polygon_rgb.points, dtype=np.float32)
    kinect_cords = np.array(
        generate_cords(size=(kinect_color_processor.height, kinect_color_processor.width)),
        dtype=np.float32)

    detected_cords_ordered = np.float32(ordering_points(kinect_cords, detected_cords))
    focus_matrix_rgb = cv2.getPerspectiveTransform(detected_cords_ordered, kinect_cords)

    ############################ TEST MATRIX
    kinect_color_processor.warp_perspective(warp_matrix=focus_matrix_rgb)
    kinect_infrared_processor.warp_perspective(warp_matrix=focus_matrix_ir)
    kinect_depth_processor.warp_perspective(warp_matrix=focus_matrix_ir)

    # Definir un umbral para considerar los valores como errores (cercanos a cero)
    threshold = 1000  # Este valor puede ser ajustado según los datos de la imagen

    # Crear una máscara donde los valores menores al umbral se consideran errores
    mask = kinect_depth_processor.image > threshold

    # Coordenadas de los puntos válidos
    valid_points = np.array(np.nonzero(mask)).T
    valid_values = kinect_depth_processor.image[mask]

    # Coordenadas de los puntos a interpolar
    invalid_points = np.array(np.nonzero(~mask)).T

    # Interpolación usando los puntos válidos
    interpolated_values = griddata(valid_points, valid_values, invalid_points, method='nearest')

    # Crear una imagen de profundidad filtrada
    filtered_depth_image = kinect_depth_processor.image.copy()
    filtered_depth_image[~mask] = interpolated_values

    kinect_depth_processor.update(filtered_depth_image)
    kinect_depth_processor.overwrite()

    kinect_depth_processor.normalize()
    kinect_depth_processor.transform_dtype()
    kinect_depth_processor.degaussing()
    kinect_depth_processor.apply_colormap()

    kinect_infrared_processor.normalize()
    kinect_infrared_processor.transform_dtype()

    principal_screen.create_window(window_name="Color Focused", image=kinect_color_processor.image)
    principal_screen.create_window(window_name="Depth Focused", image=kinect_depth_processor.image)
    principal_screen.create_window(window_name="Infrared Focused", image=kinect_infrared_processor.image)

    while (principal_screen.check_if_window_active(window_name="Infrared Focused") or
           principal_screen.check_if_window_active(window_name="Depth Focused") or
           principal_screen.check_if_window_active(window_name="Color Focused")
    ):
        time.sleep(1)

    ##########################################

    # SECOND, MOVE PROJECTOR TO SAME POSITION POINTS
    app_move_projector = instantiate_move_projector_interface(kinect=kinect, rgb_points=app_draw_polygon_rgb.points,
                                                              projector_window=projector_window,
                                                              principal_screen=principal_screen)
    detected_cords = np.array(app_move_projector.points, dtype=np.float32)
    projector_cords = np.array(
        generate_cords(size=(projector_screen.height_resolution, projector_screen.width_resolution)),
        dtype=np.float32)

    detected_cords_ordered = np.float32(ordering_points(projector_cords, detected_cords))
    focus_matrix_projector = cv2.getPerspectiveTransform(projector_cords, detected_cords_ordered)

    ############################ TEST PROJECT IMAGE
    kinect_depth_processor.resize(width=projector_screen.width_resolution, height=projector_screen.height_resolution)
    img_projected = cv2.warpPerspective(kinect_depth_processor.image, focus_matrix_projector,
                                        (projector_screen.width_resolution, projector_screen.height_resolution))

    projector_window.update_image(image=img_projected)

    # Crear mallas de coordenadas
    x, y = np.meshgrid(np.arange(kinect_depth_processor.width), np.arange(kinect_depth_processor.height))

    # Modificar las coordenadas de la malla según la profundidad
    x_new = x + kinect_depth_processor.image * 10  # El factor puede ajustarse según la profundidad
    y_new = y + kinect_depth_processor.image * 10  # El factor puede ajustarse según la profundidad

    # Remapear la imagen 2D utilizando la nueva malla de coordenadas
    map_x = np.float32(x_new)
    map_y = np.float32(y_new)

    remapped_image = cv2.remap(kinect_depth_processor.image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    projector_window.update_image(image=remapped_image)

    while projector_screen.check_if_window_active(window_name="FullScreen Projector"):
        time.sleep(1)

    principal_screen.create_window(window_name="Color Focused", image=kinect_color_processor.image)
    principal_screen.create_window(window_name="Depth Focused", image=kinect_depth_processor.image)
    principal_screen.create_window(window_name="Infrared Focused", image=kinect_infrared_processor.image)

    while (principal_screen.check_if_window_active(window_name="Infrared Focused") or
           principal_screen.check_if_window_active(window_name="Depth Focused") or
           principal_screen.check_if_window_active(window_name="Color Focused")
    ):
        time.sleep(1)

    ##################################


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
        # calibrate_sandbox_old(kinect=kinect, principal_screen=principal_screen, projector_screen=projector_screen)
        calibrate_kinect_sandbox(kinect=kinect, kinect_frame=KinectFrames.COLOR, principal_screen=principal_screen,
                                 projector_screen=projector_screen)
        calibrate_kinect_sandbox(kinect=kinect, kinect_frame=KinectFrames.INFRARED, principal_screen=principal_screen,
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
