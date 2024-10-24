import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from screen_controller.ScreenController import ScreenController


class TestScreenController(unittest.TestCase):

    @patch('screen_controller.ScreenController.generate_cords')
    def test_initialization_with_position_and_resolution(self, mock_generate_cords):
        mock_generate_cords.return_value = [(0, 0), (100, 100)]
        position = (0, 0)
        width_resolution = 100
        height_resolution = 100

        controller = ScreenController(position=position, width_resolution=width_resolution,
                                      height_resolution=height_resolution)

        self.assertEqual(controller.position, position)
        self.assertEqual(controller.width_resolution, width_resolution)
        self.assertEqual(controller.height_resolution, height_resolution)
        self.assertEqual(controller.screen_cords, [(0, 0), (100, 100)])

    def test_initialization_without_position_and_resolution(self):
        controller = ScreenController()

        self.assertIsNone(controller.position)
        self.assertEqual(controller.screen_name, "Default")
        self.assertIsNone(controller.width_resolution)
        self.assertIsNone(controller.height_resolution)
        self.assertIsNone(controller.screen_cords)

    # TODO CHECK THIS TEST
    @patch('window_controller.WindowController')
    def test_create_window(self, MockWindowController):
        mock_window_instance = MagicMock()
        MockWindowController.return_value = mock_window_instance
        controller = ScreenController(screen_name="TestScreen", width_resolution=800, height_resolution=600)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        window = controller.create_window(window_name="TestWindow", image=image)

        MockWindowController.assert_called_with(
            window_name="TestWindow",
            image=image,
            width=400,
            height=300,
            position=None,
            fullscreen=False
        )
        mock_window_instance.start.assert_called_once()
        self.assertIn("TestWindow", controller.active_window)
        self.assertEqual(controller.active_window["TestWindow"], mock_window_instance)
        window.close_window()

    @patch('window_controller.WindowController')
    def test_update_existing_window(self, MockWindowController):
        mock_window = MockWindowController.return_value
        controller = ScreenController(screen_name="TestScreen", width_resolution=800, height_resolution=600)
        controller.active_window["TestWindow"] = mock_window

        controller.update_window(window_name="TestWindow", width=400, height=300)

        mock_window.update_window.assert_called_with(width=400, height=300, position=None, fullscreen=False)

    @patch('window_controller.WindowController')
    def test_update_window_image(self, MockWindowController):
        mock_window = MockWindowController.return_value
        controller = ScreenController(screen_name="TestScreen", width_resolution=800, height_resolution=600)
        controller.active_window["TestWindow"] = mock_window
        image = MagicMock()

        controller.update_window_image(window_name="TestWindow", image=image)

        mock_window.update_image.assert_called_with(image=image)

    def test_get_window(self):
        controller = ScreenController()
        mock_window = MagicMock()
        controller.active_window["TestWindow"] = mock_window

        result = controller.get_window(window_name="TestWindow")

        self.assertEqual(result, mock_window)

    def test_remove_window(self):
        controller = ScreenController()
        controller.active_window["TestWindow"] = MagicMock()

        controller.remove_window(window_name="TestWindow")

        self.assertNotIn("TestWindow", controller.active_window)

    def test_close_window(self):
        controller = ScreenController()
        mock_window = MagicMock()
        controller.active_window["TestWindow"] = mock_window

        result = controller.close_window(window_name="TestWindow")

        mock_window.close_window.assert_called_once()
        self.assertNotIn("TestWindow", controller.active_window)
        self.assertTrue(result)

    def test_close_windows(self):
        controller = ScreenController()
        mock_window1 = MagicMock()
        mock_window2 = MagicMock()
        controller.active_window["TestWindow1"] = mock_window1
        controller.active_window["TestWindow2"] = mock_window2

        controller.close_windows()

        mock_window1.close_window.assert_called_once()
        mock_window2.close_window.assert_called_once()
        self.assertEqual(controller.active_window, {})


if __name__ == '__main__':
    unittest.main()
