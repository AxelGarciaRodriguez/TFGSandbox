import tkinter as tk
from copy import deepcopy
from tkinter import messagebox, Scale, HORIZONTAL

import numpy as np
from PIL import Image, ImageTk


def instantiate_select_depth_interface(image, principal_screen, window_size=(1920, 1080)):
    calibrate_window = tk.Tk()
    calibrate_window.geometry(
        f"{window_size[0] + 30}x{window_size[1] + 90}+{principal_screen.position[0]}+{principal_screen.position[1]}")
    app = SelectDepthInterface(window=calibrate_window, image=image, window_size=window_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class SelectDepthInterface:
    def __init__(self, window, image, window_size=(1920, 1080)):
        # Necessary variables
        self.original_image = image
        self.image = deepcopy(image)
        self.window_size = window_size

        self.minimum_depth = None
        self.maximum_depth = None

        # Initiate window
        self.window = window
        self.window.title("Depth Image Kinect")

        # Instantiate Canvas
        self.canvas = tk.Canvas(window, width=window_size[0], height=window_size[1])
        self.canvas.pack()

        # Read Image and create image object
        self.photo = None
        self.photo_obj = None
        self.update_image()

        # Sliders
        self.slider_min = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="Profundidad Mínima",
                                command=self.segment_depth_image)
        self.slider_min.set(0)
        self.slider_min.pack()

        self.slider_max = Scale(window, from_=0, to=255, orient=HORIZONTAL, label="Profundidad Máxima",
                                command=self.segment_depth_image)
        self.slider_max.set(255)
        self.slider_max.pack()

        self.accept_button = tk.Button(window, text="Aceptar", command=self.accept_after)
        self.accept_button.pack()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_interrupted = False

    def on_closing(self):
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la ventana? El proceso terminará."):
            self.process_interrupted = True
            self.window.destroy()

    def update_image(self):
        self.image = Image.fromarray(self.image)
        self.image = self.image.resize(self.window_size)
        self.photo = ImageTk.PhotoImage(self.image)
        if self.photo_obj:
            self.canvas.itemconfig(self.photo_obj, image=self.photo)
        else:
            self.photo_obj = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def segment_depth_image(self, *args):
        depth_array = np.array(deepcopy(self.original_image))
        self.minimum_depth = self.slider_min.get()
        self.maximum_depth = self.slider_max.get()

        mask = (depth_array >= self.minimum_depth) & (depth_array <= self.maximum_depth)
        segmented_array = np.where(mask, depth_array, 0)
        self.image = segmented_array
        self.update_image()

    def accept(self):
        self.window.destroy()

    def accept_after(self):
        self.window.after(0, self.accept())
