# region Kinect Control
import enum

import cv2

from kinect_module import PyKinectV2

KINECT_MAX_CHECKS_CONNECTION = 5
KINECT_SECONDS_BETWEEN_CHECK_CONNECTION = 5


class KinectFrames(enum.Enum):
    COLOR = PyKinectV2.FrameSourceTypes_Color
    DEPTH = PyKinectV2.FrameSourceTypes_Depth
    INFRARED = PyKinectV2.FrameSourceTypes_Infrared


# endregion

# region Window Control
WINDOW_MAX_RETRIES_CREATION = 3
WINDOW_SECONDS_BETWEEN_CREATIONS = 1
# endregion

# region Kinect/Projector Calibration
CALIBRATE_PATTERN_IMAGES = [KinectFrames.COLOR.name, KinectFrames.INFRARED.name]

OBJ_POINTS_KEY = "obj_points"
IMG_POINTS_KEY = "img_points"
IMG_SHAPE_KEY = "img_shape"
CAMERA_CALIBRATION_VARIABLE = "camera_matrix"
CAMERA_DISTORTION_VARIABLE = "coef_distorsion"
CAMERA_ROTATION_VARIABLE = "rvecs"
CAMERA_TRANSLATION_VARIABLE = "tvecs"

FOCUS_HOMOGRAPHY_VARIABLE = "focus_homography"
FOCUS_INV_HOMOGRAPHY_VARIABLE = "focus_inverse_homography"
FOCUS_CORDS_VARIABLE = "focus_cords"
FOCUS_CORDS_ORIGINAL_VARIABLE = "focus_original_cords"
FOCUS_DIMENSION_ORIGINAL_VARIABLE = "focus_original_dimension"
MIN_DEPTH_VARIABLE = "min_depth"
MAX_DEPTH_VARIABLE = "max_depth"

KINECT_CALIBRATION_PATH = "calibration_files\\kinect_calibration\\"
KINECT_CALIBRATION_FILENAME = "kinect_calibration.npz"
PROJECTOR_CALIBRATION_PATH = "calibration_files\\projector_calibration\\"
PROJECTOR_CALIBRATION_FILENAME = "projector_calibration.npz"

IMAGE_KINECT_SAVE_PATH = "calibration_images\\kinect_images\\"
IMAGE_PROJECTOR_SAVE_PATH = "calibration_images\\projector_images\\"

# endregion

# region Projector Calibration Literals
PATTERN_MOVE_SCALAR = 15
PATTERN_RESIZE_SCALAR = 0.1
# endregion

# region General Images
IMAGE_BASE_PATH = "base_images"
PATTERN_IMAGE_NAME = "chess_pattern.jpg"
NOT_FOUND_IMAGE_NAME = "not_found_image.png"
# endregion

# region DEPTH management (mm)
BOX_HEIGHT = 250
STANDARD_MIN_DEPTH = 500
STANDARD_MAX_DEPTH = 3000
CONTOURS_EPSILON_FACTOR = 0.000001
CONTOURS_MIN_AREA = 500


class ConfigControllerEnum(enum.Enum):
    MIN_DEPTH = STANDARD_MIN_DEPTH
    MAX_DEPTH = STANDARD_MAX_DEPTH
    CONTOURS_LEVEL_STEPS = 10
    ERRORS_UMBRAL = 5
    MEDIUM_NOISE = 15
    BIG_NOISE = 30
    NO_SENSE_CHANGES = 80
    COLORMAP = cv2.COLORMAP_JET
    RESET_IMAGE = False


class ConfigControllerNamesEnum(enum.Enum):
    MIN_DEPTH = "Profundidad Mínima (mm)"
    MAX_DEPTH = 'Profundidad máxima (mm)'
    CONTOURS_LEVEL_STEPS = 'Niveles de contorno'
    ERRORS_UMBRAL = 'Umbral de error'
    MEDIUM_NOISE = 'Ruido medio'
    BIG_NOISE = 'Ruido alto'
    COLORMAP = 'Mapa de color'


class ConfigControllerSliderEnum(enum.Enum):
    CONTOURS_LEVEL_STEPS = [1, 30]
    ERRORS_UMBRAL = [0, 30]
    MEDIUM_NOISE = [15, 45]
    BIG_NOISE = [30, 80]

# endregion
