import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from utils import transform_cords, get_depth_information


def instantiate_draw_polygon_interface(image_rgb, image_depth, window_size=(960, 540)):
    calibrate_window = tk.Tk()
    app = DrawPolygonInterface(window=calibrate_window, image_rgb=image_rgb, image_depth=image_depth,
                               window_size=window_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class DrawPolygonInterface:
    def __init__(self, window, image_rgb, image_depth, window_size=(960, 540)):
        # Necessary variables
        self.image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        self.image_depth = image_depth
        self.window_size = window_size

        image_rgb_shape = image_rgb.shape[:2]
        self.initial_height, self.initial_width = image_rgb_shape
        self.points = []
        self.points_depth = []
        self.points_draw = []
        self.polygon = None

        # Initiate window
        self.window = window
        self.window.title("Proyección Kinect")

        # Instantiate Canvas
        self.canvas = tk.Canvas(window, width=window_size[0], height=window_size[1])
        self.canvas.pack()

        # Read Image and create image object
        self.update_image()

        # Bind buttons
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.delete_polygon)

        self.accept_button = tk.Button(window, text="Aceptar", command=self.accept_polygon_after)
        self.accept_button.pack()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_interrupted = False

    def on_closing(self):
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la ventana? El proceso terminará."):
            self.process_interrupted = True
            self.window.destroy()

    def add_point(self, event):
        if len(self.points) < 4:
            self.points_draw.append((event.x, event.y))
            x_image = self.canvas.canvasx(event.x) * self.initial_width / self.canvas.winfo_width()
            y_image = self.canvas.canvasy(event.y) * self.initial_height / self.canvas.winfo_height()

            x_transformed, y_transformed = transform_cords(point=(x_image, y_image),
                                                           original_size=self.image_rgb.shape[:2],
                                                           new_size=self.image_depth.shape[:2])
            z_information = get_depth_information(depth_image=self.image_depth, point=(x_transformed, y_transformed))
            self.points.append((x_image, y_image))
            self.points_depth.append((x_image, y_image, z_information))

            self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="red")
            self.canvas.create_text(event.x, event.y - 10, text=f"({int(x_image)}, {int(y_image)}, {int(z_information)})", fill="black", font=('Helvetica 10 bold'))
            if len(self.points) > 1:
                self.canvas.create_line(self.points_draw[-2][0], self.points_draw[-2][1], event.x, event.y, fill="blue")
            if len(self.points) == 4:
                self.polygon = self.canvas.create_polygon(self.points_draw, outline="green", fill="")

    def delete_polygon(self, event):
        self.points.clear()
        self.points_draw.clear()
        for obj in self.canvas.find_all():
            if obj != self.photo_obj:
                self.canvas.delete(obj)

    def update_image(self):
        self.image = Image.fromarray(self.image_rgb)
        self.image = self.image.resize(self.window_size)
        self.photo = ImageTk.PhotoImage(self.image)
        self.photo_obj = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def accept_polygon(self):
        self.window.destroy()

    def accept_polygon_after(self):
        self.window.after(0, self.accept_polygon())
