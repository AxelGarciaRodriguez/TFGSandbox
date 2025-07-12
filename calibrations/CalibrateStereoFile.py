import cv2

from calibrations.CalibrationFile import CalibrationClass


class CalibrationStereoClass:
    def __init__(self, calibration_file_camera_1: CalibrationClass, calibration_file_camera_2: CalibrationClass,
                 calibration_path_file=None):
        # FOCUS
        self.calibration_file_camera_1 = calibration_file_camera_1
        self.calibration_file_camera_2 = calibration_file_camera_2

        # FILES
        self.calibration_path_file = calibration_path_file

    def calculate_stereo_matrix(self, rgb_image, ir_image):
        # Calibración estéreo
        obj_points = self.calibration_file_camera_1.obj_points

        flags = cv2.CALIB_FIX_INTRINSIC  # o usa otras combinaciones de flags según lo anterior
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            self.calibration_file_camera_1.img_points,
            self.calibration_file_camera_2.img_points,
            self.calibration_file_camera_1.camera_matrix,
            self.calibration_file_camera_1.cof_distortion,
            self.calibration_file_camera_2.camera_matrix,
            self.calibration_file_camera_2.cof_distortion,
            self.calibration_file_camera_1.image_shape[::-1],
            flags=flags
        )

        # Rectificación estéreo
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.calibration_file_camera_1.camera_matrix,
            self.calibration_file_camera_1.cof_distortion,
            self.calibration_file_camera_2.camera_matrix,
            self.calibration_file_camera_2.cof_distortion,
            self.calibration_file_camera_1.image_shape[::-1], R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )

        # Mapas de remapeo
        map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(
            self.calibration_file_camera_1.camera_matrix,
            self.calibration_file_camera_1.cof_distortion, R1, P1,
            self.calibration_file_camera_1.image_shape[::-1],
            cv2.CV_32FC1
        )

        map1_ir, map2_ir = cv2.initUndistortRectifyMap(
            self.calibration_file_camera_2.camera_matrix,
            self.calibration_file_camera_2.cof_distortion, R2, P2,
            self.calibration_file_camera_2.image_shape[::-1],
            cv2.CV_32FC1
        )

        # Aplicación de los mapas
        rectified_rgb = cv2.remap(rgb_image, map1_rgb, map2_rgb, cv2.INTER_LINEAR)
        rectified_ir = cv2.remap(ir_image, map1_ir, map2_ir, cv2.INTER_LINEAR)

        total_error = 0
        for i in range(len(obj_points)):
            imgpoints2_rgb, _ = cv2.projectPoints(obj_points[i], self.calibration_file_camera_1.camera_rotation[i],
                                                  self.calibration_file_camera_1.camera_translation[i],
                                                  self.calibration_file_camera_1.camera_matrix,
                                                  self.calibration_file_camera_1.cof_distortion)
            imgpoints2_rgb = imgpoints2_rgb.reshape(-1, 2)
            error_rgb = cv2.norm(self.calibration_file_camera_1.img_points[i], imgpoints2_rgb, cv2.NORM_L2) / len(imgpoints2_rgb)

            imgpoints2_ir, _ = cv2.projectPoints(obj_points[i], self.calibration_file_camera_2.camera_rotation[i],
                                                  self.calibration_file_camera_2.camera_translation[i],
                                                  self.calibration_file_camera_2.camera_matrix,
                                                  self.calibration_file_camera_2.cof_distortion)
            imgpoints2_ir = imgpoints2_ir.reshape(-1, 2)
            error_ir = cv2.norm(self.calibration_file_camera_2.img_points[i], imgpoints2_ir, cv2.NORM_L2) / len(imgpoints2_ir)

            total_error += error_rgb + error_ir

        print("Error de reproyección promedio:", total_error / len(obj_points))

        return rectified_rgb, rectified_ir