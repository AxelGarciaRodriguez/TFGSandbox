import tkinter as tk
from copy import deepcopy
from tkinter import messagebox, Scale

import cv2
import numpy as np
from PIL import Image, ImageTk


def instantiate_select_color_interface(image, principal_screen, window_size=(1920, 1080)):
    calibrate_window = tk.Tk()
    calibrate_window.geometry(
        f"{window_size[0] + 30}x{window_size[1] + 90}+{principal_screen.position[0]}+{principal_screen.position[1]}")
    app = SelectColorInterface(window=calibrate_window, image=image, window_size=window_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class SelectColorInterface:
    def __init__(self, window, image, window_size=(1920, 1080)):
        # Necessary variables
        self.original_image = image
        self.image = deepcopy(image)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.window_size = window_size

        # Initiate window
        self.window = window
        self.window.title("Color Image Kinect")

        # Read Image and create image object
        self.photo = None
        self.photo_obj = None

        self.image_frame = tk.Frame(window)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)

        self.update_image()

        # SLIDERS
        self.h_lower_slider = Scale(window, from_=0, to=179, orient='horizontal', command=self.segment_color_image)
        self.s_lower_slider = Scale(window, from_=0, to=255, orient='horizontal', command=self.segment_color_image)
        self.v_lower_slider = Scale(window, from_=0, to=255, orient='horizontal', command=self.segment_color_image)
        self.h_upper_slider = Scale(window, from_=0, to=179, orient='horizontal', command=self.segment_color_image)
        self.s_upper_slider = Scale(window, from_=0, to=255, orient='horizontal', command=self.segment_color_image)
        self.v_upper_slider = Scale(window, from_=0, to=255, orient='horizontal', command=self.segment_color_image)

        self.h_lower_slider.set(0)
        self.s_lower_slider.set(100)
        self.v_lower_slider.set(100)
        self.h_upper_slider.set(10)
        self.s_upper_slider.set(255)
        self.v_upper_slider.set(255)

        h_lower_label = tk.Label(window, text="H Lower")
        s_lower_label = tk.Label(window, text="S Lower")
        v_lower_label = tk.Label(window, text="V Lower")
        h_upper_label = tk.Label(window, text="H Upper")
        s_upper_label = tk.Label(window, text="S Upper")
        v_upper_label = tk.Label(window, text="V Upper")

        h_lower_label.pack(padx=5, pady=5)
        self.h_lower_slider.pack(padx=5, pady=5)
        s_lower_label.pack(padx=5, pady=5)
        self.s_lower_slider.pack(padx=5, pady=5)
        v_lower_label.pack(padx=5, pady=5)
        self.v_lower_slider.pack(padx=5, pady=5)
        h_upper_label.pack(padx=5, pady=5)
        self.h_upper_slider.pack(padx=5, pady=5)
        s_upper_label.pack(padx=5, pady=5)
        self.s_upper_slider.pack(padx=5, pady=5)
        v_upper_label.pack(padx=5, pady=5)
        self.v_upper_slider.pack(padx=5, pady=5)

        self.accept_button = tk.Button(window, text="Aceptar", command=self.accept_after)
        self.accept_button.pack()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_interrupted = False

    def on_closing(self):
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la ventana? El proceso terminará."):
            self.process_interrupted = True
            self.window.destroy()

    def update_image(self):
        segmented_image_pil = Image.fromarray(self.image)
        segmented_image_tk = ImageTk.PhotoImage(segmented_image_pil)

        self.image_label.config(image=segmented_image_tk)
        self.image_label.image = segmented_image_tk

    def segment_color_image(self, *args):
        h_lower = self.h_lower_slider.get()
        s_lower = self.s_lower_slider.get()
        v_lower = self.v_lower_slider.get()
        h_upper = self.h_upper_slider.get()
        s_upper = self.s_upper_slider.get()
        v_upper = self.v_upper_slider.get()

        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])

        mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)

        self.image = deepcopy(self.original_image)

        segmented_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        self.image = segmented_image_rgb
        self.update_image()

    def accept(self):
        self.window.destroy()

    def accept_after(self):
        self.window.after(0, self.accept())
