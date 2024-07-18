import os
from itertools import permutations

import numpy as np


def generate_cords(size, initial_position=None):
    if not initial_position:
        initial_position = (0, 0)
    height, width = size

    return [initial_position, (initial_position[0], initial_position[1] + height),
            (initial_position[0] + width, initial_position[1]),
            (initial_position[0] + width, initial_position[1] + height)]


def transform_cords(point, original_size, new_size):
    original_height, original_width = original_size
    new_height, new_width = new_size

    x_position = point[0] * new_width / original_width
    y_position = point[1] * new_height / original_height

    return int(x_position), int(y_position)


def get_depth_information(depth_image, point):
    return depth_image[point[1], point[0]]


def generate_relative_path(path_parts):
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, *path_parts)
    image_path = os.path.normpath(relative_path)
    return image_path


def ordering_points(l_point1, l_point2):
    l_distance_ordered = []
    for l_ordering in permutations(list(range(len(l_point1)))):
        d = 0
        l_ordered_point = []
        for i1, i2 in enumerate(l_ordering):
            d += np.linalg.norm(np.array(l_point1[i1])
                                - np.array(l_point2[i2]))
            l_ordered_point += [l_point2[i2]]

        l_distance_ordered += [[d, l_ordered_point]]

    l_ordered_point = min(l_distance_ordered, key=lambda x: x[0])[1]

    return l_ordered_point
