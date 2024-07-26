# region Kinect Control
KINECT_MAX_CHECKS_CONNECTION = 5
KINECT_SECONDS_BETWEEN_CHECK_CONNECTION = 5
# endregion

# region Window Control
WINDOW_MAX_RETRIES_CREATION = 3
WINDOW_SECONDS_BETWEEN_CREATIONS = 1
# endregion

# region Kinect/Projector Calibration
NUMBER_DEPTH_IMAGES_FOR_DEPTH_CALIBRATION = 100

RGB_IMAGES_KEY = "rgb"
DEPTH_IMAGES_KEY = "depth"
DEPTH_NP_KEY = "depth_np"
IR_IMAGES_KEY = "infrared"

CALIBRATE_PATTERN_IMAGES = [RGB_IMAGES_KEY, IR_IMAGES_KEY]

OBJ_POINTS_KEY = "obj_points"
IMG_POINTS_KEY = "img_points"
CAMERA_CALIBRATION_VARIABLE = "camera_matrix"
CAMERA_DISTORTION_VARIABLE = "coef_distorsion"
CAMERA_ROTATION_VARIABLE = "rvecs"
CAMERA_TRANSLATION_VARIABLE = "tvecs"

FOCUS_HOMOGRAPHY_VARIABLE = "focus_homography"
FOCUS_INV_HOMOGRAPHY_VARIABLE = "focus_inverse_homography"
FOCUS_CORDS_VARIABLE = "focus_cords"
FOCUS_CORDS_ORIGINAL_VARIABLE = "focus_original_cords"
FOCUS_DIMENSION_ORIGINAL_VARIABLE = "focus_original_dimension"

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
