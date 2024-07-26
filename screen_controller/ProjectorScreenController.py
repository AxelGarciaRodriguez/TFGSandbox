from calibrations.CalibrationFile import CalibrationClass
from literals import PROJECTOR_CALIBRATION_PATH, PROJECTOR_CALIBRATION_FILENAME
from screen_controller.ScreenController import ScreenController
from utils import generate_relative_path


class ProjectorScreenController(ScreenController):

    def __init__(self, position=None, screen_name=None, width_resolution=None, height_resolution=None):
        super().__init__(position=position, screen_name=screen_name, width_resolution=width_resolution,
                         height_resolution=height_resolution)

        calibration_path = generate_relative_path([PROJECTOR_CALIBRATION_PATH, PROJECTOR_CALIBRATION_FILENAME])
        self.calibration = CalibrationClass(calibration_path_file=calibration_path)
        self.calibration.read_calibration()
