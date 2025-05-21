import tkinter as tk
from copy import deepcopy
from tkinter import messagebox

from PIL import Image, ImageTk

from image_management.ImageObject import ImageObject
from image_management.ImageGenerator import ImageGenerator
from image_management.ImageTransformerBase import ImageTransformerBase
from kinect_controller.KinectController import KinectFrames


def instantiate_move_projector_interface(kinect, rgb_points, projector_window, principal_screen):
    window_size = (principal_screen.width_resolution - 25, principal_screen.height_resolution - 90)
    calibrate_window = tk.Tk()
    calibrate_window.geometry(
        f"{window_size[0]}x{window_size[1]}+{principal_screen.position[0]}+{principal_screen.position[1]}")
    window_size = (principal_screen.width_resolution - 30, principal_screen.height_resolution - 130)
    app = MoveProjectorPointsInterface(window=calibrate_window, kinect=kinect, rgb_points=rgb_points,
                                       projector_window=projector_window, window_size=window_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class MoveProjectorPointsInterface:
    def __init__(self, window, kinect, rgb_points, projector_window, window_size=(960, 540),
                 displacement_value=5):

        # IMPORTANT VARIABLES
        self.points = []
        self.kinect_points = rgb_points
        self.kinect = kinect
        self.projector_window = projector_window
        self.displacement_value = displacement_value
        self.window_size = window_size

        self.image_processor_kinect = None
        self.image_processor_projector = None

        self.actual_point_kinect = None
        self.actual_point_projector = None
        self.actual_index = 0

        # Initialize window
        self.window = window
        self.window.title("Calibración Proyector")

        # Instantiate canvas
        self.canvas = tk.Canvas(window, width=window_size[0], height=window_size[1])
        self.canvas.pack()

        # CREATE BINDS AND ACCEPT BUTTON
        self.window.bind("<KeyPress-Up>", self.move_up)
        self.window.bind("<KeyPress-Down>", self.move_down)
        self.window.bind("<KeyPress-Left>", self.move_left)
        self.window.bind("<KeyPress-Right>", self.move_right)

        self.accept_button = tk.Button(window, text="Siguiente", command=self.accept_point)
        self.accept_button.pack()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_interrupted = False

        # GET KINECT IMAGE
        self.get_kinect_image()
        self.generate_projector_image()

        # SHOW KINECT IMAGE
        self.photo = None
        self.photo_obj = None
        self.show_kinect_image()

        # GENERATE KINECT POINT IN CANVAS
        self.point = None
        self.get_kinect_point()
        self.draw_kinect_point()

        # GENERATE PROJECTOR POINT
        self.generate_project_point()
        self.draw_projector_point()

        # UPDATE IMAGE BACKGROUND LOOP
        self.update_background_loop()

    def on_closing(self):
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la ventana? El proceso terminará."):
            self.process_interrupted = True
            self.window.destroy()

    def get_kinect_point(self):
        self.actual_point_kinect = self.kinect_points[self.actual_index]

    def draw_kinect_point(self):
        # CANVAS
        kinect_point_scaled = (
            (self.actual_point_kinect[0] / self.image_processor_kinect.width * self.window_size[0]),
            (self.actual_point_kinect[1] / self.image_processor_kinect.height * self.window_size[1])
        )
        self.canvas.delete(self.point)

        self.point = self.canvas.create_oval(kinect_point_scaled[0] - 2, kinect_point_scaled[1] - 2,
                                             kinect_point_scaled[0] + 2, kinect_point_scaled[1] + 2,
                                             fill="red")
        self.canvas.tag_raise(self.point)

    def generate_project_point(self):
        self.actual_point_projector = (
            self.image_processor_projector.height // 2, self.image_processor_projector.width // 2
        )

    def draw_projector_point(self):
        # IMAGE BASE
        self.image_processor_projector.restore()
        image = ImageTransformerBase.draw_point(image=self.image_processor_projector.image,
                                                point=self.actual_point_projector, radius=3, color=(255, 255, 255))
        self.image_processor_projector.update(image=image)
        self.projector_window.update_image(image=self.image_processor_projector.image)

    def save_project_point(self):
        self.points.append(deepcopy(self.actual_point_projector))

    def update_background_loop(self):
        if self.kinect.check_if_new_image(kinect_frame=KinectFrames.COLOR):
            image_kinect = self.kinect.get_image_calibrate(kinect_frame=KinectFrames.COLOR, avoid_camera_focus=True)
            self.image_processor_kinect = ImageObject(image=image_kinect)
            self.show_kinect_image()

        self.window.after(30, self.update_background_loop)

    def get_kinect_image(self):
        if self.kinect.check_if_new_image(kinect_frame=KinectFrames.COLOR):
            image_kinect = self.kinect.get_image(kinect_frame=KinectFrames.COLOR)
            self.image_processor_kinect = ImageObject(image=image_kinect)

    def generate_projector_image(self):
        background_color_image = ImageGenerator.generate_color_image(shape=self.projector_window.resolution)
        self.image_processor_projector = ImageObject(image=background_color_image)

    def show_kinect_image(self):
        image = Image.fromarray(self.image_processor_kinect.image[..., ::-1])
        image = image.resize(self.window_size)
        self.photo = ImageTk.PhotoImage(image)
        if self.photo_obj:
            self.canvas.itemconfig(self.photo_obj, image=self.photo)
        else:
            self.photo_obj = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def accept_point(self):
        # SAVE PROJECTOR POINT
        self.save_project_point()

        # UPDATE ACTUAL INDEX
        self.actual_index += 1

        if self.actual_index == len(self.kinect_points):
            messagebox.showinfo("All points calibrated", "Todos los puntos han sido calibrados.")
            self.accept_polygon_after()
        else:
            # GET NEXT KINECT POINT
            self.get_kinect_point()
            self.draw_kinect_point()

            # GENERATE NEW PROJECTOR POINT
            self.generate_project_point()
            self.draw_projector_point()

    def move_up(self, event):
        self.actual_point_projector = (
            self.actual_point_projector[0], self.actual_point_projector[1] - self.displacement_value)
        self.draw_projector_point()

    def move_down(self, event):
        self.actual_point_projector = (
            self.actual_point_projector[0], self.actual_point_projector[1] + self.displacement_value)
        self.draw_projector_point()

    def move_left(self, event):
        self.actual_point_projector = (
            self.actual_point_projector[0] - self.displacement_value, self.actual_point_projector[1])
        self.draw_projector_point()

    def move_right(self, event):
        self.actual_point_projector = (
            self.actual_point_projector[0] + self.displacement_value, self.actual_point_projector[1])
        self.draw_projector_point()

    def accept_polygon(self):
        self.window.destroy()

    def accept_polygon_after(self):
        self.window.after(0, self.accept_polygon())
