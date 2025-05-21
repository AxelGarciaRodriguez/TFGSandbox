import cv2
import numpy as np


class ImageGenerator:

    @staticmethod
    def generate_color_image(shape, color=(0, 0, 0)):
        image = np.full((shape[1], shape[0], 3), color, dtype=np.uint8)
        return image

    @staticmethod
    def generate_image_with_other_image(image, position, image_shape, shape, background_color=(0, 0, 0)):
        background_image = ImageGenerator.generate_color_image(shape=shape, color=background_color)
        resized_image = cv2.resize(image, (image_shape[1], image_shape[0]))

        x, y = position
        resized_height, resized_width = resized_image.shape[:2]
        height, width = background_image.shape[:2]

        start_x = max(x, 0)
        start_y = max(y, 0)
        end_x = min(x + resized_width, width)
        end_y = min(y + resized_height, height)

        img_start_x = 0 if x >= 0 else -x
        img_start_y = 0 if y >= 0 else -y
        img_end_x = resized_width if x + resized_width <= width else width - x
        img_end_y = resized_height if y + resized_height <= height else height - y

        background_image[start_y:end_y, start_x:end_x] = resized_image[img_start_y:img_end_y, img_start_x:img_end_x]

        return background_image

    # TODO ADD MORE METHODS FOR IMAGE GENERATION
