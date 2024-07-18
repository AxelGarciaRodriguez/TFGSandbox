import logging

from utils import generate_cords
from window_controller.WindowController import WindowController


class ScreenController(object):

    def __init__(self, position=None, screen_name=None, width_resolution=None, height_resolution=None):
        self.position = position if position else (0, 0)
        self.screen_name = screen_name if screen_name else "Default"
        self.width_resolution = width_resolution if width_resolution else None
        self.height_resolution = height_resolution if height_resolution else None
        self.screen_resolution = (self.width_resolution, self.height_resolution)
        if position and width_resolution and height_resolution:
            self.screen_cords = generate_cords(initial_position=self.position,
                                               size=(height_resolution, width_resolution))
        else:
            self.screen_cords = None

        self.active_window = {}

    def create_window(self, window_name, image, width=None, height=None, position=None, fullscreen=False):
        if fullscreen:
            width = self.width_resolution
            height = self.height_resolution
        elif not width or not height:
            width = self.width_resolution // 2
            height = self.height_resolution // 2

        if not position:
            position = self.position

        # Check if window already exists
        already_created_window = self.get_window(window_name=window_name)
        if already_created_window:
            already_created_window.update_window(width=width, height=height, position=position, fullscreen=fullscreen)
            already_created_window.update_image(image=image)
            return already_created_window
        else:
            logging.debug(f"Creating window '{window_name}' in '{self.screen_name}' screen ...")
            new_window = WindowController(window_name=window_name, image=image, width=width, height=height,
                                          position=position, fullscreen=fullscreen)
            new_window.start()
            self.active_window[window_name] = new_window
            return new_window

    def update_window(self, window_name, width=None, height=None, position=None, fullscreen=False):
        already_created_window = self.get_window(window_name=window_name)
        if already_created_window:
            already_created_window.update_window(width=width, height=height, position=position, fullscreen=fullscreen)
            return already_created_window
        logging.debug(f"Window {window_name} does not exist in {self.screen_name}, cannot be updated")
        self.remove_window(window_name=window_name)
        return None

    def update_window_image(self, window_name, image):
        already_created_window = self.get_window(window_name=window_name)
        if already_created_window:
            already_created_window.update_image(image=image)
            return already_created_window

        logging.warning(f"Window {window_name} does not exist in {self.screen_name}, image cannot be updated")
        self.remove_window(window_name=window_name)
        return None

    def get_window(self, window_name):
        if window_name in self.active_window.keys():
            return self.active_window[window_name]
        return None

    def remove_window(self, window_name):
        if window_name in self.active_window.keys():
            self.active_window.pop(window_name)

    def close_window(self, window_name):
        if window_name in self.active_window.keys():
            self.active_window[window_name].close_window()
            self.active_window.pop(window_name)
            return True
        else:
            logging.debug(f"Window {window_name} not found in screen {self.screen_name}")
            self.active_window.pop(window_name)
            return False

    def close_windows(self):
        for active_window in self.active_window.values():
            active_window.close_window()
        self.active_window.clear()

    def check_if_window_active(self, window_name):
        if window_name in self.active_window.keys():
            return self.active_window[window_name].check_if_alive()
        else:
            return False
