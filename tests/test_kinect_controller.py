import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from kinect_controller.KinectController import KinectController, KinectFrames


class TestKinectController(unittest.TestCase):

    @patch('kinect_controller.KinectController.PyKinectRuntime')
    @patch('kinect_controller.KinectController.KINECT_MAX_CHECKS_CONNECTION', 3)
    @patch('kinect_controller.KinectController.KINECT_SECONDS_BETWEEN_CHECK_CONNECTION', 0.1)
    def test_initialization(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        mock_kinect_instance.has_new_color_frame.return_value = True

        controller = KinectController([KinectFrames.COLOR])

        self.assertIsNotNone(controller.kinect)
        self.assertEqual(controller.kinect, mock_kinect_instance)

    @patch('kinect_controller.KinectController.PyKinectRuntime')
    def test_check_if_new_image(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        mock_kinect_instance.has_new_color_frame.return_value = True
        mock_kinect_instance.has_new_depth_frame.return_value = False

        controller = KinectController([KinectFrames.COLOR])

        self.assertTrue(controller.check_if_new_image(KinectFrames.COLOR))
        self.assertFalse(controller.check_if_new_image(KinectFrames.DEPTH))

    @patch('kinect_controller.KinectController.PyKinectRuntime')
    def test_get_frame(self, MockPyKinectRuntime):
        mock_kinect_instance = MagicMock()
        MockPyKinectRuntime.return_value = mock_kinect_instance

        mock_kinect_instance.has_new_color_frame.return_value = True
        mock_kinect_instance.get_last_color_frame.return_value = np.array([1, 2, 3, 4])

        controller = KinectController([KinectFrames.COLOR])

        frame = controller.get_frame(KinectFrames.COLOR)
        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(frame, np.array([1, 2, 3, 4]))

    @patch('kinect_controller.KinectController.PyKinectRuntime')
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


if __name__ == '__main__':
    unittest.main()
