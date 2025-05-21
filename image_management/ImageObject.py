from image_management.ImageTransformerBase import ImageTransformerBase


class ImageObject:
    def __init__(self, image=None, image_absolute_path=None, image_transform_class=ImageTransformerBase):
        self.image_transform_class = image_transform_class

        if image_absolute_path is not None:
            self.image = self.image_transform_class.load(image_path=image_absolute_path)
        elif image is not None:
            self.image = image
        else:
            raise AttributeError("ImageObject class needs at least an image path or an image object")

        self.initial_image = self.image_transform_class.duplicate(image=self.image)
        self.image_shape = None
        self.width, self.height = None, None
        self.update_image_information()

    def update_image_information(self):
        self.image_shape = self.image_transform_class.get_image_shape(image=self.image)
        self.width, self.height = self.image_transform_class.get_image_width_and_height(image=self.image)

    # region Basic Operations
    def update(self, image):
        self.image = image
        self.update_image_information()

    def overwrite(self):
        self.initial_image = self.image_transform_class.duplicate(image=self.image)

    def restore(self):
        image = self.image_transform_class.duplicate(image=self.initial_image)
        self.update(image=image)

    def save(self, output_path):
        self.image_transform_class.save(image=self.image, output_path=output_path)

    def load(self, image_path):
        image = self.image_transform_class.load(image_path=image_path)
        self.update(image=image)

    # endregion
