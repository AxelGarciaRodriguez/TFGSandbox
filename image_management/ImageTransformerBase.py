from copy import deepcopy

import cv2
import numpy as np


class ImageTransformerBase:

    # region Basic image info
    @staticmethod
    def get_image_shape(image):
        return image.shape[:2]

    @staticmethod
    def get_image_width(image):
        _, width = ImageTransformerBase.get_image_shape(image=image)
        return width

    @staticmethod
    def get_image_height(image):
        height, _ = ImageTransformerBase.get_image_shape(image=image)
        return height

    @staticmethod
    def get_image_width_and_height(image):
        height, width = ImageTransformerBase.get_image_shape(image=image)
        return width, height

    # endregion

    # region Basic operations with images
    @staticmethod
    def duplicate(image):
        return deepcopy(image)

    @staticmethod
    def save(image, output_path):
        cv2.imwrite(output_path, image)

    @staticmethod
    def load(image_path):
        return cv2.imread(image_path)

    # endregion

    # region Basic transformations
    @staticmethod
    def resize(image, width=None, height=None):
        if width is None and height is None:
            return image

        image_width, image_height = ImageTransformerBase.get_image_width_and_height(image=image)

        if width is None:
            ratio = height / float(image_height)
            dimension = (int(image_width * ratio), height)
        elif height is None:
            ratio = width / float(image_width)
            dimension = (width, int(image_height * ratio))
        else:
            dimension = (width, height)

        image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def rotate(image, angle):
        width, height = ImageTransformerBase.get_image_width_and_height(image=image)
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height))
        return image

    # endregion

    # region Apply matrix (distort and warp)
    @staticmethod
    def distort(image, camera_matrix, distortion_coefficients):
        return cv2.undistort(image, camera_matrix, distortion_coefficients)

    @staticmethod
    def warp_perspective(image, warp_matrix, output_size=None):
        if not output_size:
            width, height = ImageTransformerBase.get_image_width_and_height(image=image)
            output_size = (width, height)

        return cv2.warpPerspective(image, warp_matrix, output_size)

    # endregion

    # region Filter Image
    @staticmethod
    def apply_mask(image, condition, value):
        return np.where(condition, value, image)

    @staticmethod
    def get_mask_between_values(image, min_value, max_value, mode=cv2.THRESH_BINARY):
        _, mask = cv2.threshold(image, min_value, max_value, mode)
        return mask

    @staticmethod
    def get_masks_by_steps(image, step_value, mode=cv2.THRESH_BINARY):
        masks = []
        for threshold in range(np.min(image), np.max(image), step_value):
            masks.append(ImageTransformerBase.get_mask_between_values(image=image, min_value=threshold,
                                                                      max_value=threshold + step_value, mode=mode))
        return masks

    @staticmethod
    def remove_zeros(image, radius=5, flags=cv2.INPAINT_TELEA):
        mask = (image == 0).astype(np.uint8)
        return cv2.inpaint(image.astype(np.float32), mask, inpaintRadius=radius, flags=flags)

    @staticmethod
    def degaussing(image, ksize=(5, 5), sigma_x=0):
        return cv2.GaussianBlur(image, ksize, sigma_x)

    @staticmethod
    def normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
        return cv2.normalize(src=image, dst=None, alpha=alpha, beta=beta, norm_type=norm_type)

    @staticmethod
    def transform_dtype(image, dtype=np.uint8):
        return dtype(image)

    @staticmethod
    def apply_colormap(image, colormap=cv2.COLORMAP_JET):
        return cv2.applyColorMap(image, colormap)

    @staticmethod
    def change_color(image, color=cv2.COLOR_BGR2GRAY):
        return cv2.cvtColor(image, color)

    @staticmethod
    def change_to_binary_colors(image, thresh=128, maxval=255):
        _, binary_img = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
        return binary_img

    @staticmethod
    def gamma_correction(image, gamma=1.5, dtype='uint8'):
        return np.array(255 * (image / 255) ** gamma, dtype=dtype)

    @staticmethod
    def invert(image):
        return image[..., ::-1]

    # endregion

    # region Search information in image
    @staticmethod
    def find_chessboard_corners(image, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE):
        gray_image = ImageTransformerBase.change_color(image=image, color=cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image=gray_image, patternSize=pattern_size, flags=flags)
        return ret, corners

    @staticmethod
    def calculate_sub_pix_corner(image, corners,
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        gray_image = ImageTransformerBase.change_color(image=image, color=cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
        return corners

    @staticmethod
    def find_contours(image, mode=cv2.RETR_TREE, flags=cv2.CHAIN_APPROX_SIMPLE):
        contours, _ = cv2.findContours(image, mode, flags)
        return contours

    @staticmethod
    def get_contour_area(contour):
        return cv2.contourArea(contour)

    # endregion

    # region Draw
    @staticmethod
    def draw_point(image, point, color=(0, 255, 0), radius=5):
        return cv2.circle(image, tuple(point), radius, color, -1)

    @staticmethod
    def draw_line(image, line, color=(0, 0, 255), thickness=2):
        return cv2.line(image, tuple(line[0]), tuple(line[1]), color, thickness)

    @staticmethod
    def draw_points(image, points, color=(0, 255, 0), radius=5):
        for point in points:
            image = ImageTransformerBase.draw_point(image=image, point=point, color=color, radius=radius)
        return image

    @staticmethod
    def draw_lines(image, lines, color=(0, 0, 255), thickness=2):
        for line in lines:
            image = ImageTransformerBase.draw_line(image=image, line=line, color=color, thickness=thickness)
        return image

    @staticmethod
    def draw_polygon(image, points, color=(255, 0, 0), thickness=2, close_polygon=True):
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        return cv2.polylines(image, [pts], close_polygon, color, thickness)

    @staticmethod
    def draw_chessboard_corners(image, pattern_size, corners):
        return cv2.drawChessboardCorners(image=image, patternSize=pattern_size, corners=corners, patternWasFound=True)

    @staticmethod
    def draw_contours(image, contours, color=(255, 0, 0), thickness=2):
        return cv2.drawContours(image, contours, -1, color, thickness)

    @staticmethod
    def approx_poly(points, epsilon_factor=0.01, closed_poly=True):
        epsilon = epsilon_factor * cv2.arcLength(points, closed_poly)
        return cv2.approxPolyDP(points, epsilon, closed_poly)

    # endregion

    # region Instantiate Specific Methods
    @staticmethod
    def remove_data_between_distance_neighbors(image, min_depth, max_depth, other_image=None, kernel_shape=(3, 3),
                                               iterations=10):
        raise NotImplementedError("Method 'remove_data_between_distance_neighbors' not defined")

    @staticmethod
    def remove_data_between_distance(image, min_depth, max_depth):
        raise NotImplementedError("Method 'remove_data_between_distance' not defined")

    @staticmethod
    def set_data_between_distance(image, min_depth, max_depth):
        raise NotImplementedError("Method 'set_data_between_distance' not defined")

    @staticmethod
    def normalize_between_distance(image, min_depth, max_depth):
        raise NotImplementedError("Method 'normalize_between_distance' not defined")

    # endregion
