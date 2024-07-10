import logging
import threading
import time
import cv2

from literals import WINDOW_MAX_RETRIES_CREATION, WINDOW_SECONDS_BETWEEN_CREATIONS
from kinect_controller.KinectLock import lock


class WindowController(threading.Thread):
    def __init__(self, window_name, image, width=None, height=None, position=None, fullscreen=False):
        super(WindowController, self).__init__()
        self.window_name = window_name
        self.image = image
        self.width = width
        self.height = height
        self.position = position
        self.fullscreen = fullscreen

        # Image Management
        self.image_changed = True

        # Window Management
        self.stopped = False

    def run(self):
        # Lock creation window process
        with lock:
            for attempt in range(WINDOW_MAX_RETRIES_CREATION):
                try:
                    self.create_window()
                    self.configure_window()
                    # First image show
                    self.image_changed = False
                    self.show_image(self.image)
                except Exception as e:
                    logging.warning(f"Attempt {attempt + 1} creating window {self.window_name} failed: {e}")
                    if attempt + 1 == WINDOW_MAX_RETRIES_CREATION:
                        logging.error(f"Failed to create window after {WINDOW_MAX_RETRIES_CREATION} attempts")
                        raise e
                    time.sleep(WINDOW_SECONDS_BETWEEN_CREATIONS)

        while not self.stopped:
            if self.image_changed:
                self.image_changed = False
                with lock:
                    # Lock image updates
                    self.show_image(self.image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.close_window()

        cv2.destroyWindow(self.window_name)

    def create_window(self):
        if self.fullscreen:
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        logging.debug(f"Window '{self.window_name}' created successfully")

    def configure_window(self):
        if self.width and self.height:
            self.set_window_size(self.width, self.height)
        if self.position:
            self.set_window_position(*self.position)
        if self.fullscreen:
            self.set_fullscreen()

    def set_window_size(self, width, height):
        logging.debug(f"Setting window size to {width}x{height}")
        cv2.resizeWindow(self.window_name, width, height)

    def set_window_position(self, x, y):
        logging.debug(f"Setting window position to ({x}, {y})")
        cv2.moveWindow(self.window_name, x, y)

    def set_fullscreen(self):
        logging.debug("Setting window to fullscreen")
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def validate_inputs(self):
        assert isinstance(self.window_name, str), "Window name must be a string"
        assert self.image is not None, "Image cannot be None"
        if self.width is not None:
            assert isinstance(self.width, int) and self.width > 0, "Width must be a positive integer"
        if self.height is not None:
            assert isinstance(self.height, int) and self.height > 0, "Height must be a positive integer"
        if self.position is not None:
            assert isinstance(self.position, tuple) and len(
                self.position) == 2, "Position must be a tuple with two elements"

    def start(self):
        self.validate_inputs()
        super().start()

    def show_image(self, image):
        cv2.imshow(self.window_name, image)

    def update_image(self, image):
        self.image = image
        self.image_changed = True

    def close_window(self):
        self.stopped = True

    def update_window(self, width=None, height=None, position=None, fullscreen=False):
        self.width = width
        self.height = height
        self.position = position
        self.fullscreen = fullscreen

        self.validate_inputs()
        self.configure_window()

    # region Window Functions
    def add_mouse_callback_function(self, function, params):
        cv2.setMouseCallback(self.window_name, function, params)

    # endregion
