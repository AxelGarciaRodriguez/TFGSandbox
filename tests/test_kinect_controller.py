import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from kinect_controller.KinectController import KinectController, KinectFrames


class TestKinectController(unittest.TestCase):

    @patch('kinect_module.PyKinectRuntime.PyKinectRuntime')
    def test_initialization(self, MockPyKinectRuntime):
        mock_kinect_instance = MockPyKinectRuntime.return_value
        mock_kinect_instance.has_new_color_frame.return_value = True

        # Inicializar el controlador Kinect
        controller = KinectController([KinectFrames.COLOR])

        # Verificar que el atributo kinect del controlador no sea None
        self.assertIsNotNone(controller.kinect)

        # Verificar que el objeto kinect del controlador sea el mismo que el mock
        self.assertEqual(controller.kinect, mock_kinect_instance)

    @patch('kinect_module.PyKinectRuntime.PyKinectRuntime')
    def test_check_if_new_image(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        mock_kinect_instance.has_new_color_frame.return_value = True
        mock_kinect_instance.has_new_depth_frame.return_value = False

        controller = KinectController([KinectFrames.COLOR])

        self.assertTrue(controller.check_if_new_image(KinectFrames.COLOR))
        self.assertFalse(controller.check_if_new_image(KinectFrames.DEPTH))

    @patch('kinect_module.PyKinectRuntime.PyKinectRuntime')
    def test_get_image(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        mock_kinect_instance.has_new_color_frame.return_value = True
        mock_kinect_instance.get_last_color_frame.return_value = np.zeros((1080 * 1920 * 4,))
        mock_kinect_instance.color_frame_desc.Height = 1080
        mock_kinect_instance.color_frame_desc.Width = 1920

        controller = KinectController([KinectFrames.COLOR])

        image = controller.get_image(KinectFrames.COLOR)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (1080, 1920, 4))

    @patch('kinect_module.PyKinectRuntime')
    def test_display_rgb_image(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        # Mock the color frame data
        mock_color_frame = np.random.randint(0, 256, (1080 * 1920 * 4,), dtype=np.uint8)
        mock_kinect_instance.has_new_color_frame.return_value = True
        mock_kinect_instance.get_last_color_frame.return_value = mock_color_frame
        mock_kinect_instance.color_frame_desc.Height = 1080
        mock_kinect_instance.color_frame_desc.Width = 1920

        controller = KinectController([KinectFrames.COLOR])

        image = controller.get_image(KinectFrames.COLOR)

        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (1080, 1920, 4))

        # Display the image using OpenCV
        cv2.imshow('Color Image', image)
        cv2.waitKey(0)  # Wait for a key press to close the image window
        cv2.destroyAllWindows()

    @patch('kinect_module.PyKinectRuntime')
    def test_display_depth_image(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        # Mock the depth frame data
        mock_depth_frame = np.random.randint(0, 65536, (424 * 512,), dtype=np.uint16)  # Example depth frame data
        mock_kinect_instance.has_new_depth_frame.return_value = True
        mock_kinect_instance.get_last_depth_frame.return_value = mock_depth_frame
        mock_kinect_instance.depth_frame_desc.Height = 424
        mock_kinect_instance.depth_frame_desc.Width = 512

        controller = KinectController([KinectFrames.DEPTH])

        image = controller.get_image(KinectFrames.DEPTH)

        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (424, 512))

        # Scale the depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.03), cv2.COLORMAP_JET)

        # Display the image using OpenCV
        cv2.imshow('Depth Image', depth_colormap)
        cv2.waitKey(0)  # Wait for a key press to close the image window
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
