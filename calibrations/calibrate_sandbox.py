import argparse
import logging
import sys
import time

import cv2
import numpy as np

from image_manager.ImageBase import ImageBase
from image_manager.ImageGenerator import ImageGenerator
from image_manager.ImageProcessor import ImageProcessor
from interfaces.CalibrateKinectProjectorInterface import instantiate_calibrate_interface
from interfaces.DrawPolygonInterface import instantiate_draw_polygon_interface
from interfaces.SelectorScreenInterface import selector_screens
from kinect_controller.KinectController import KinectFrames, KinectController
from literals import NOT_FOUND_IMAGE_NAME, IMAGE_BASE_PATH
from utils import generate_cords, ordering_points, transform_cords, generate_relative_path


def get_args():
    parser = argparse.ArgumentParser(prog="calibrate_sandbox",
                                     description='Initialize calibrations for sandbox detection',
                                     epilog='',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--logging', help='Logging level (DEBUG=10, INFO=20, WARNING=30, ERROR=40 or CRITICAL=50)',
                        type=int, required=False, default=logging.INFO)

    args, unknown = parser.parse_known_args()
    return args

def nothing(x):
    pass

def calibrate_sandbox(kinect, principal_screen, projector_screen):
    # FIRST, SELECT SAND BOX IN KINECT IMAGE
    background_color_image = ImageGenerator.generate_color_image(shape=projector_screen.screen_resolution)
    projector_window = projector_screen.create_window(window_name="FullScreen Projector", image=background_color_image, fullscreen=True)

    kinect_image_rgb = kinect.get_image(kinect_frame=KinectFrames.COLOR)
    kinect_image_depth = kinect.get_image(kinect_frame=KinectFrames.DEPTH)

    app_draw_polygon = instantiate_draw_polygon_interface(image_rgb=kinect_image_rgb, image_depth=kinect_image_depth)

    detected_cords = np.array(app_draw_polygon.points, dtype=np.float32)
    kinect_cords = np.array(
        generate_cords(size=(kinect.kinect.color_frame_desc.Height, kinect.kinect.color_frame_desc.Width)),
        dtype=np.float32)

    detected_cords_ordered = np.float32(ordering_points(kinect_cords, detected_cords))
    focus_matrix_rgb = cv2.getPerspectiveTransform(detected_cords_ordered, kinect_cords)

    detected_cords_depth_transform = []
    for cord in detected_cords_ordered:
        x_transform, y_trasnform = transform_cords(cord, original_size=kinect_image_rgb.shape[:2],
                                                   new_size=kinect_image_depth.shape[:2])
        detected_cords_depth_transform.append((x_transform, y_trasnform))
    detected_cords_depth_transform = np.array(detected_cords_depth_transform, dtype=np.float32)

    kinect_cords = np.array(
        generate_cords(size=(kinect.kinect.depth_frame_desc.Height, kinect.kinect.depth_frame_desc.Width)),
        dtype=np.float32)
    focus_matrix_depth = cv2.getPerspectiveTransform(detected_cords_depth_transform, kinect_cords)

    ############################ TEST MATRIX
    kinect_rgb_processor = ImageProcessor(image=kinect_image_rgb)
    kinect_depth_processor = ImageProcessor(image=kinect_image_depth)

    kinect_rgb_processor.warp_perspective(warp_matrix=focus_matrix_rgb, output_size=(
    kinect.kinect.color_frame_desc.Width, kinect.kinect.color_frame_desc.Height))
    kinect_depth_processor.warp_perspective(warp_matrix=focus_matrix_depth, output_size=(
    kinect.kinect.depth_frame_desc.Width, kinect.kinect.depth_frame_desc.Height))

    kinect_depth_processor.normalize()
    kinect_depth_processor.transform_dtype()
    kinect_depth_processor.apply_colormap()

    principal_screen.create_window(window_name="RGB Focused", image=kinect_rgb_processor.image)
    principal_screen.create_window(window_name="Depth Focused", image=kinect_depth_processor.image)

    while principal_screen.check_if_window_active(window_name="RGB Focused") or principal_screen.check_if_window_active(
            window_name="Depth Focused"):
        time.sleep(1)

    ################################################################
    not_found_image = ImageBase(
        image_absolute_path=generate_relative_path([IMAGE_BASE_PATH, NOT_FOUND_IMAGE_NAME])).image
    app_projector = instantiate_calibrate_interface(kinect=kinect, projector_window=projector_window,
                                                    projector_screen=projector_screen,
                                                    previous_images={}, pattern_image=kinect_depth_processor.image,
                                                    not_found_image=not_found_image)


    ###############################################################

    cv2.namedWindow('Control de Profundidad')

    # Crear trackbars para el control del rango de profundidad
    cv2.createTrackbar('Depth Min', 'Control de Profundidad', 0, 255, nothing)
    cv2.createTrackbar('Depth Max', 'Control de Profundidad', 255, 255, nothing)

    # Leer la imagen de profundidad
    kinect_depth_processor.restore()
    kinect_depth_processor.warp_perspective(warp_matrix=focus_matrix_depth, output_size=(
    kinect.kinect.depth_frame_desc.Width, kinect.kinect.depth_frame_desc.Height))
    kinect_depth_processor.normalize()
    imagen_profundidad = kinect_depth_processor.image

    while True:
        # Obtener los valores actuales de las trackbars
        depth_min = cv2.getTrackbarPos('Depth Min', 'Control de Profundidad')
        depth_max = cv2.getTrackbarPos('Depth Max', 'Control de Profundidad')

        # Definir el rango de profundidad basado en los valores de las trackbars
        min_depth = depth_min
        max_depth = depth_max

        # Crear una máscara con el rango de profundidad definido
        mascara = cv2.inRange(imagen_profundidad, min_depth, max_depth)

        # Aplicar operaciones morfológicas para mejorar la segmentación
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        # Convertir la imagen de profundidad a una imagen en escala de grises para visualizar
        imagen_profundidad_color = cv2.applyColorMap(cv2.convertScaleAbs(imagen_profundidad, alpha=0.03),
                                                     cv2.COLORMAP_JET)

        # Mostrar resultados
        cv2.imshow('Imagen de Profundidad', imagen_profundidad_color)
        cv2.imshow('Imagen de Color', kinect_rgb_processor.image)
        cv2.imshow('Máscara', mascara)

        # Presionar 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

    # Crear una ventana
    cv2.namedWindow('Control de HSV')

    # Crear trackbars para el control del rango HSV
    cv2.createTrackbar('Hue Min', 'Control de HSV', 0, 179, nothing)
    cv2.createTrackbar('Hue Max', 'Control de HSV', 179, 179, nothing)
    cv2.createTrackbar('Sat Min', 'Control de HSV', 0, 255, nothing)
    cv2.createTrackbar('Sat Max', 'Control de HSV', 255, 255, nothing)
    cv2.createTrackbar('Val Min', 'Control de HSV', 0, 255, nothing)
    cv2.createTrackbar('Val Max', 'Control de HSV', 255, 255, nothing)

    # Leer la imagen
    imagen_rgb = kinect_rgb_processor.image

    while True:
        # Convertir de BGR a HSV
        imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2HSV)

        # Obtener los valores actuales de las trackbars
        h_min = cv2.getTrackbarPos('Hue Min', 'Control de HSV')
        h_max = cv2.getTrackbarPos('Hue Max', 'Control de HSV')
        s_min = cv2.getTrackbarPos('Sat Min', 'Control de HSV')
        s_max = cv2.getTrackbarPos('Sat Max', 'Control de HSV')
        v_min = cv2.getTrackbarPos('Val Min', 'Control de HSV')
        v_max = cv2.getTrackbarPos('Val Max', 'Control de HSV')

        # Definir el rango de color HSV basado en los valores de las trackbars
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])

        # Crear una máscara con el rango de color definido
        mascara = cv2.inRange(imagen_hsv, lower_hsv, upper_hsv)

        # Aplicar operaciones morfológicas para mejorar la segmentación
        kernel = np.ones((5, 5), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        # Mostrar la máscara y la imagen original
        cv2.imshow('Imagen Original', imagen_rgb)
        cv2.imshow('Máscara', mascara)

        # Presionar 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

    # GET SAND POSITION
    imagen_hsv = cv2.cvtColor(kinect_rgb_processor.image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([20, 40, 40])
    upper_hsv = np.array([30, 255, 255])

    mascara = cv2.inRange(imagen_hsv, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imagen_con_contornos = cv2.drawContours(kinect_rgb_processor.image.copy(), contornos, -1, (0, 255, 0), 3)
    cv2.imshow('Imagen Original', kinect_rgb_processor.image)
    cv2.imshow('Máscara', mascara)
    cv2.imshow('Contornos de Arena', imagen_con_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Leer la imagen de profundidad
    imagen_profundidad = cv2.imread('ruta_a_tu_imagen_profundidad.png', cv2.IMREAD_UNCHANGED)

    # Definir el rango de profundidad para detectar arena (ajustar estos valores según tu imagen)
    min_depth = 100  # Valor mínimo de profundidad para la arena
    max_depth = 200  # Valor máximo de profundidad para la arena

    # Crear una máscara con el rango de profundidad definido
    mascara = cv2.inRange(kinect_depth_processor.image, min_depth, max_depth)

    # Aplicar operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convertir la imagen de profundidad a una imagen en escala de grises para visualizar
    imagen_profundidad_color = cv2.applyColorMap(cv2.convertScaleAbs(imagen_profundidad, alpha=0.03), cv2.COLORMAP_JET)
    imagen_con_contornos = cv2.drawContours(imagen_profundidad_color, contornos, -1, (0, 255, 0), 3)

    # Mostrar resultados
    cv2.imshow('Imagen de Profundidad', imagen_profundidad_color)
    cv2.imshow('Máscara', mascara)
    cv2.imshow('Contornos de Arena', imagen_con_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # SECOND, MOVE PROJECTOR TO SAME POSITION POINTS


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
        calibrate_sandbox(kinect=kinect, principal_screen=principal_screen, projector_screen=projector_screen)

    except Exception as error:
        logging.error(f"Error in manual calibrations: {error}")
    finally:
        principal_screen.close_windows()
        projector_screen.close_windows()
        kinect.close()


if __name__ == '__main__':
    main()
