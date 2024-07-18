# region Kinect Control
KINECT_MAX_CHECKS_CONNECTION = 5
KINECT_SECONDS_BETWEEN_CHECK_CONNECTION = 3
# endregion

# region Window Control
WINDOW_MAX_RETRIES_CREATION = 3
WINDOW_SECONDS_BETWEEN_CREATIONS = 1
# endregion

# region Kinect-Projector Calibration Literals
IMAGE_BASE_PATH = "base_images"
PATTERN_IMAGE_NAME = "chess_pattern.jpg"
NOT_FOUND_IMAGE_NAME = "not_found_image.png"

KINECT_CALIBRATION_PATH = "calibration_files\\kinect_calibration\\"
KINECT_CALIBRATION_FILENAME = "kinect_calibration.npz"
KINECT_CAMERA_CALIBRATION_VARIABLE = "camera_matrix"
KINECT_CAMERA_DISTORSION_VARIABLE = "coef_distorsion"
KINECT_CAMERA_ROTATION_VARIABLE = "rvecs"
KINECT_CAMERA_TRASLATION_VARIABLE = "tvecs"

PROJECTOR_CALIBRATION_PATH = "calibration_files\\projector_calibration\\"
PROJECTOR_CALIBRATION_FILENAME = "projector_calibration.npz"
PROJECTOR_CAMERA_CALIBRATION_VARIABLE = "camera_matrix"
PROJECTOR_CAMERA_DISTORSION_VARIABLE = "coef_distorsion"
PROJECTOR_CAMERA_ROTATION_VARIABLE = "rvecs"
PROJECTOR_CAMERA_TRASLATION_VARIABLE = "tvecs"

IMAGE_KINECT_SAVE_PATH = "calibration_images\\kinect_images\\"
IMAGE_PROJECTOR_SAVE_PATH = "calibration_images\\projector_images\\"
RGB_IMAGES_KEY = "rgb"
DEPTH_IMAGES_KEY = "depth"
PROJECTED_IMAGES_KEY = "projected"

PATTERN_MOVE_SCALAR = 15
PATTERN_RESIZE_SCALAR = 0.1
# endregion
