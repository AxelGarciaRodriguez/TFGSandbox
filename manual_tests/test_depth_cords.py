import cv2

from kinect_controller.KinectController import KinectController
from literals import KinectFrames

img_rgb = None
img_depth = None
height_rgb, width_rgb = None, None
height_depth, width_depth = None, None


def on_mouse_event_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        show_rgb_point(x, y)


def on_mouse_event_depth(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        show_depth_point(x, y)


def show_rgb_point(x_rgb, y_rgb):
    x_depth = int(x_rgb * (width_depth / width_rgb))
    y_depth = int(y_rgb * (height_depth / height_rgb))

    img_rgb_copy = img_rgb.copy()
    img_depth_copy = img_depth.copy()

    cv2.circle(img_rgb_copy, (x_rgb, y_rgb), 5, (0, 0, 255), -1)
    cv2.circle(img_depth_copy, (x_depth, y_depth), 5, (0, 255, 0), -1)

    # Mostrar las imágenes actualizadas
    cv2.imshow('Imagen RGB', img_rgb_copy)
    cv2.imshow('Imagen de Profundidad', img_depth_copy)


def show_depth_point(x_depth, y_depth):
    x_rgb = int(x_depth * (width_rgb / width_depth))
    y_rgb = int(y_depth * (height_rgb / height_depth))

    img_rgb_copy = img_rgb.copy()
    img_depth_copy = img_depth.copy()

    cv2.circle(img_depth_copy, (x_depth, y_depth), 5, (0, 255, 0), -1)  # Punto verde en profundidad
    cv2.circle(img_rgb_copy, (x_rgb, y_rgb), 5, (0, 0, 255), -1)  # Punto rojo en RGB

    cv2.imshow('Imagen RGB', img_rgb_copy)
    cv2.imshow('Imagen de Profundidad', img_depth_copy)


# Cargar las imágenes RGB y de profundidad
kinect = KinectController(kinect_frames=[KinectFrames.COLOR, KinectFrames.DEPTH, KinectFrames.INFRARED])

img_depth = kinect.get_image(kinect_frame=KinectFrames.DEPTH)
img_depth = cv2.convertScaleAbs(img_depth, alpha=255 / 4500)
img_rgb = kinect.get_image(kinect_frame=KinectFrames.COLOR)

height_rgb, width_rgb = img_rgb.shape[:2]
height_depth, width_depth = img_depth.shape[:2]

cv2.imshow('Imagen RGB', img_rgb)
cv2.imshow('Imagen de Profundidad', img_depth)

cv2.setMouseCallback('Imagen RGB', on_mouse_event_rgb)
cv2.setMouseCallback('Imagen de Profundidad', on_mouse_event_depth)

cv2.waitKey(0)
cv2.destroyAllWindows()
