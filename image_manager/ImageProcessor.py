import cv2
import numpy as np

from image_manager.ImageBase import ImageBase


class ImageProcessor(ImageBase):
    def __init__(self, image=None, image_absolute_path=None):
        super().__init__(image=image, image_absolute_path=image_absolute_path)

    # region Transforms

    def undistort(self, camera_matrix, distortion_coefficients):
        image = cv2.undistort(self.image, camera_matrix, distortion_coefficients)
        self.update(image=image)
        return self.image

    def warp_perspective(self, warp_matrix, output_size=None):
        if not output_size:
            output_size = (self.width, self.height)
        image = cv2.warpPerspective(self.image, warp_matrix, output_size)
        self.update(image=image)
        return self.image

    # endregion

    # region Filter Image

    def remove_zeros(self):
        mask = (self.image == 0).astype(np.uint8)
        image = cv2.inpaint(self.image.astype(np.float32), mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        self.update(image=image)
        return self.image

    def degaussing(self, ksize=(5, 5), sigmaX=0):
        image = cv2.GaussianBlur(self.image, ksize, sigmaX)
        self.update(image=image)
        return self.image

    def normalize(self, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
        normalized_image = cv2.normalize(src=self.image, dst=None, alpha=alpha, beta=beta, norm_type=norm_type)
        self.update(image=normalized_image)
        return self.image

    def transform_dtype(self, dtype=np.uint8):
        image = dtype(self.image)
        self.update(image=image)
        return self.image

    def apply_colormap(self, colormap=cv2.COLORMAP_JET):
        image = cv2.applyColorMap(self.image, colormap)
        self.update(image=image)
        return self.image

    def change_color(self, color=cv2.COLOR_BGR2GRAY):
        image = cv2.cvtColor(self.image, color)
        self.update(image=image)
        return self.image

    def change_to_binary_colors(self, thresh=128, maxval=255):
        _, binary_img = cv2.threshold(self.image, thresh, maxval, cv2.THRESH_BINARY)
        self.update(image=binary_img)
        return self.image

    def gamma_correction(self, gamma=1.5):
        gamma_corrected = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
        self.update(image=gamma_corrected)
        return self.image

    # endregion

    # region Search information in image

    def find_chessboard_corners(self, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE):
        gray_image = self.change_color(color=cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image=gray_image, patternSize=pattern_size, flags=flags)
        # if ret:
        #     self.restore()
        #     image = self.draw_chessboard_corners(pattern_size=pattern_size, corners=corners)
        #     cv2.imshow('Chessboard corners', image)
        #     cv2.waitKey(500)

        self.restore()
        return ret, corners

    def calculate_sub_pix_corner(self, corners,
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        gray_image = self.change_color(color=cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
        self.restore()
        return corners

    # endregion

    # region Draw

    def draw_point(self, point, color=(0, 255, 0), radius=5):
        cv2.circle(self.image, tuple(point), radius, color, -1)
        return self.image

    def draw_line(self, line, color=(0, 0, 255), thickness=2):
        cv2.line(self.image, tuple(line[0]), tuple(line[1]), color, thickness)
        return self.image

    def draw_points(self, points, color=(0, 255, 0), radius=5):
        for point in points:
            self.draw_point(point=point, color=color, radius=radius)
        return self.image

    def draw_lines(self, lines, color=(0, 0, 255), thickness=2):
        for line in lines:
            self.draw_line(line=line, color=color, thickness=thickness)
        return self.image

    def draw_polygon(self, points, color=(255, 0, 0), thickness=2, close_polygon=True):
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.image, [pts], close_polygon, color, thickness)
        return self.image

    def draw_chessboard_corners(self, pattern_size, corners):
        image = cv2.drawChessboardCorners(image=self.image, patternSize=pattern_size, corners=corners,
                                          patternWasFound=True)
        self.update(image=image)
        return self.image

    def draw_contours(self, contours, color=(255, 0, 0), thickness=2):
        image = cv2.drawContours(self.image, contours, -1, color, thickness)
        self.update(image=image)
        return self.image

    # endregion
