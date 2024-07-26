from copy import deepcopy

import cv2


class ImageBase:
    def __init__(self, image=None, image_absolute_path=None):
        if image_absolute_path is not None:
            self.image = None
            self.load(image_absolute_path)
        elif image is not None:
            self.image = image
        else:
            raise AttributeError("ImageBase class needs at least an image path or an image object")

        self.initial_image = deepcopy(self.image)
        self.image_shape = self.image.shape[:2]
        self.height, self.width = self.image.shape[:2]

    def update_image_information(self):
        self.image_shape = self.image.shape[:2]
        self.height, self.width = self.image.shape[:2]

    # region Basic Operations

    def update(self, image):
        self.image = image
        self.update_image_information()

    def overwrite(self):
        self.initial_image = deepcopy(self.image)

    def restore(self):
        image = deepcopy(self.initial_image)
        self.update(image=image)

    def save(self, output_path):
        cv2.imwrite(output_path, self.image)

    def load(self, image_path):
        image = cv2.imread(image_path)
        self.update(image=image)

    # endregion

    # region Basic transformations

    def resize(self, width=None, height=None):
        if width is None and height is None:
            return self.image

        if width is None:
            ratio = height / float(self.height)
            dimension = (int(self.width * ratio), height)
        elif height is None:
            ratio = width / float(self.width)
            dimension = (width, int(self.height * ratio))
        else:
            dimension = (width, height)

        image = cv2.resize(self.image, dimension, interpolation=cv2.INTER_AREA)
        self.update(image=image)
        return self.image

    def rotate(self, angle):
        center = (self.width / 2, self.height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(self.image, matrix, (self.width, self.height))
        self.update(image=image)
        return self.image

    def image_to_jpg(self):
        return self.image

    # endregion
