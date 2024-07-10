from screen_controller.ScreenController import ScreenController


class ProjectorScreenController(ScreenController):

    def __init__(self, position=None, screen_name=None, width_resolution=None, height_resolution=None):
        super().__init__(position=position, screen_name=screen_name, width_resolution=width_resolution,
                         height_resolution=height_resolution)
