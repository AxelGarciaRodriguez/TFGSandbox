import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
import numpy as np

from image_manager.ImageBase import ImageBase
from image_manager.ImageGenerator import ImageGenerator
from image_manager.ImageProcessor import ImageProcessor
from kinect_controller.KinectController import KinectFrames
from literals import PATTERN_MOVE_SCALAR, PATTERN_RESIZE_SCALAR, RGB_IMAGES_KEY, DEPTH_IMAGES_KEY, PROJECTED_IMAGES_KEY
from utils import transform_cords

from PIL import Image, ImageTk


def instantiate_calibrate_interface(kinect, projector_window, projector_screen, previous_images, pattern_image,
                                    not_found_image, window_image_size=(960, 540)):
    calibrate_window = tk.Tk()
    app = CalibrateKinectProjectorInterface(window=calibrate_window, kinect=kinect, projector_window=projector_window,
                                            projector_screen=projector_screen, previous_images=previous_images,
                                            pattern_image=pattern_image, not_found_image=not_found_image,
                                            window_image_size=window_image_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class CalibrateKinectProjectorInterface:

    def __init__(self, window, kinect, projector_window, projector_screen, previous_images, pattern_image,
                 not_found_image, window_image_size=(960, 540)):
        # Necessary variables
        self.kinect = kinect
        self.pattern_size = None
        self.photos_taken = previous_images  # Previous images contains 1 key for each image, and a dict for rgb, depth and patter images
        self.projector_window = projector_window
        self.projector_screen = projector_screen

        self.pattern_image = pattern_image
        if pattern_image is not None:
            self.pattern_projected_position = (0, 0)
            self.pattern_projected_size = pattern_image.shape[:2]

        self.object_points = {}
        self.image_points = {}
        self.depth_points = {}
        self.image_shape = None

        self.window_image_size = window_image_size
        self.not_found_image = not_found_image

        # Initiate window
        self.window = window
        self.window.title("Calibrate Kinect-Projector System")

        self.frame = tk.Frame(self.window)
        self.frame.pack()

        row_counter = 0

        # Generate window board size registers
        tk.Label(self.frame, text="Tamaño de la matriz (ancho x alto):").grid(row=row_counter, column=0, columnspan=2, padx=5,
                                                                              pady=5)
        self.entry_size_x = tk.Entry(self.frame)
        self.entry_size_x.grid(row=row_counter, column=2)

        tk.Label(self.frame, text="X").grid(row=0, column=3)

        self.entry_size_y = tk.Entry(self.frame)
        self.entry_size_y.grid(row=row_counter, column=4)

        if pattern_image is not None:
            row_counter += 1
            # Manage pattern projected
            self.checkbox_value = tk.BooleanVar(self.frame)
            self.checkbox_project_pattern = ttk.Checkbutton(self.frame, text="¿Proyectar patrón?",
                                                            variable=self.checkbox_value, command=self.checkbox_clicked)
            self.checkbox_project_pattern.grid(row=row_counter, column=0, columnspan=1, padx=5, pady=5)

            self.bigger_button = tk.Button(self.frame, text="+", command=self.resize_pattern_bigger)
            self.bigger_button.grid(row=row_counter, column=3)

            self.smaller_button = tk.Button(self.frame, text="-", command=self.resize_pattern_smaller)
            self.smaller_button.grid(row=row_counter, column=4)

            row_counter += 1

            # Generate controls for patron move
            tk.Label(self.frame, text="Controles patrón:").grid(row=row_counter, column=0, columnspan=1, padx=5, pady=5)

            self.left_button = tk.Button(self.frame, text="←", command=self.move_pattern_left)
            self.left_button.grid(row=row_counter, column=1)

            self.up_button = tk.Button(self.frame, text="↑", command=self.move_pattern_up)
            self.up_button.grid(row=row_counter, column=2)

            self.down_button = tk.Button(self.frame, text="↓", command=self.move_pattern_down)
            self.down_button.grid(row=row_counter, column=3)

            self.right_button = tk.Button(self.frame, text="→", command=self.move_pattern_right)
            self.right_button.grid(row=row_counter, column=4)

        row_counter += 1
        # Generate countdown number
        self.countdown_label = tk.Label(self.frame, text="")
        self.countdown_label.grid(row=row_counter, column=2, columnspan=3, padx=5, pady=5)

        # Generate image position
        self.last_image = tk.Label(self.frame)
        self.last_image.grid(row=row_counter, column=0, columnspan=5, padx=5, pady=5)

        row_counter += 1
        # Generate change photo labels
        self.current_index = 0

        self.prev_button = tk.Button(self.frame, text="Prev", command=self.prev_image)
        self.prev_button.grid(row=row_counter, column=1)

        self.index_label = tk.Label(self.frame, text=f"{self.current_index + 1} / {len(self.photos_taken)}")
        self.index_label.grid(row=row_counter, column=2)

        self.next_button = tk.Button(self.frame, text="Next", command=self.next_image)
        self.next_button.grid(row=row_counter, column=3)

        row_counter += 1
        # Generate take photo, delete photo and finish process buttons
        self.button_take_photo = tk.Button(self.frame, text="Tomar Imagen", command=self.take_image)
        self.button_take_photo.grid(row=row_counter, column=0, padx=5, pady=5)

        self.button_delete_photo = tk.Button(self.frame, text="Borrar Imagen", command=self.delete_image)
        self.button_delete_photo.grid(row=row_counter, column=1, padx=5, pady=5)

        self.button_calculate_pattern = tk.Button(self.frame, text="Calcular Patrones", command=self.calculate_patterns)
        self.button_calculate_pattern.grid(row=row_counter, column=2, padx=5, pady=5)

        self.button_delete_pattern = tk.Button(self.frame, text="Borrar Patrones", command=self.delete_patterns)
        self.button_delete_pattern.grid(row=row_counter, column=3, padx=5, pady=5)

        self.button_exit = tk.Button(self.frame, text="Terminar Calibración", command=self.exit_after)
        self.button_exit.grid(row=row_counter, column=4, padx=5, pady=5)

        self.update_image()
        if self.pattern_image is not None:
            self.update_projected_pattern()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_interrupted = False

    # region Window management

    def on_closing(self):
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la ventana? El proceso terminará."):
            self.process_interrupted = True
            self.window.destroy()

    def countdown(self, number):
        self.countdown_label.config(text=number, font=("Helvetica", 36))

    def exit(self):
        # Check if we have at least 1 image
        if not self.photos_taken:
            messagebox.showerror("Missing photos", "Se necesita al menos 1 imagen para la calibración.")
            return

        self.calculate_patterns()
        if len(self.photos_taken) != len(self.object_points):
            return

        self.window.destroy()

    def exit_after(self):
        self.window.after(0, self.exit())

    def change_status_all_buttons(self, status):
        self.prev_button.config(state=status)
        self.next_button.config(state=status)
        self.button_delete_photo.config(state=status)
        self.button_exit.config(state=status)
        self.button_calculate_pattern.config(state=status)
        self.button_delete_pattern.config(state=status)
        self.button_take_photo.config(state=status)
        if self.pattern_image is not None:
            self.up_button.config(state=status)
            self.down_button.config(state=status)
            self.left_button.config(state=status)
            self.right_button.config(state=status)
            self.smaller_button.config(state=status)
            self.bigger_button.config(state=status)

    def enable_buttons(self):
        self.change_status_all_buttons(status=tk.NORMAL)

    def disable_buttons(self):
        self.change_status_all_buttons(status=tk.DISABLED)

    def update_image(self):
        if not self.photos_taken:
            # READ BASE IMAGE
            image = Image.fromarray(self.not_found_image)
            image = image.resize(self.window_image_size)
            photo = ImageTk.PhotoImage(image)
            self.last_image.config(image=photo)
            self.last_image.image = photo

            self.prev_button.config(state=tk.DISABLED)
            self.button_delete_photo.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            self.index_label.config(text=f"0 / 0")
            return

        image_processor = list(self.photos_taken.values())[self.current_index][RGB_IMAGES_KEY]
        image = Image.fromarray(image_processor.image)
        image = image.resize(self.window_image_size)
        photo = ImageTk.PhotoImage(image)
        self.last_image.config(image=photo)
        self.last_image.image = photo

        self.index_label.config(text="{} / {}".format(self.current_index + 1, len(self.photos_taken)))
        self.prev_button.config(state=tk.NORMAL)
        self.button_delete_photo.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)

    # endregion

    # region Move/Resize pattern projected
    def move_pattern_left(self):
        self.pattern_projected_position = (self.pattern_projected_position[0] - PATTERN_MOVE_SCALAR,
                                           self.pattern_projected_position[1])
        self.update_projected_pattern()

    def move_pattern_up(self):
        self.pattern_projected_position = (self.pattern_projected_position[0],
                                           self.pattern_projected_position[1] + PATTERN_MOVE_SCALAR)
        self.update_projected_pattern()

    def move_pattern_down(self):
        self.pattern_projected_position = (self.pattern_projected_position[0],
                                           self.pattern_projected_position[1] - PATTERN_MOVE_SCALAR)
        self.update_projected_pattern()

    def move_pattern_right(self):
        self.pattern_projected_position = (self.pattern_projected_position[0] + PATTERN_MOVE_SCALAR,
                                           self.pattern_projected_position[1])
        self.update_projected_pattern()

    def resize_pattern_bigger(self):
        self.pattern_projected_size = (int(self.pattern_projected_size[0] * (1 + PATTERN_RESIZE_SCALAR)),
                                       int(self.pattern_projected_size[1] * (1 + PATTERN_RESIZE_SCALAR)))
        self.update_projected_pattern()

    def resize_pattern_smaller(self):
        self.pattern_projected_size = (int(self.pattern_projected_size[0] * (1 - PATTERN_RESIZE_SCALAR)),
                                       int(self.pattern_projected_size[1] * (1 - PATTERN_RESIZE_SCALAR)))
        self.update_projected_pattern()

    def checkbox_clicked(self):
        self.update_projected_pattern()

    def update_projected_pattern(self):
        if self.checkbox_value.get():
            # Generate new image
            new_image = ImageGenerator.generate_image_with_other_image(image=self.pattern_image,
                                                                       position=self.pattern_projected_position,
                                                                       image_shape=self.pattern_projected_size,
                                                                       shape=(self.projector_screen.width_resolution,
                                                                              self.projector_screen.height_resolution),
                                                                       background_color=(0, 0, 0))
        else:
            new_image = ImageGenerator.generate_color_image(
                shape=(self.projector_screen.width_resolution, self.projector_screen.height_resolution),
                color=(0, 0, 0))
        self.projector_window.update_image(image=new_image)

    # endregion

    # region Manage Buttons Image

    def prev_image(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.photos_taken) - 1
        self.update_image()

    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.photos_taken):
            self.current_index = 0
        self.update_image()

    def delete_image(self):
        if self.photos_taken:
            if messagebox.askokcancel("Borrar Imagen", "¿Está seguro de borrar la imagen actual?"):
                image_name = list(self.photos_taken.keys())[self.current_index]
                self.photos_taken.pop(image_name)
                self.delete_image_information(image_name=image_name)

                self.current_index = self.current_index - 1 if self.current_index > 0 else 0
                self.update_image()

    def take_image(self):
        self.disable_buttons()

        self.countdown_label.place(relx=0.5, rely=0.5, anchor="center")
        self.countdown_label.lift()
        for i in range(3, 0, -1):
            self.countdown(i)
            self.window.update()
            self.window.after(1000)

        self.countdown_label.place_forget()

        image_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
        self.photos_taken[image_name] = {}
        # GET RGB IMAGE
        image = self.kinect.get_image(kinect_frame=KinectFrames.COLOR)
        image_processor = ImageProcessor(image=image)
        self.photos_taken[image_name][RGB_IMAGES_KEY] = image_processor

        # GET DEPTH IMAGE
        image = self.kinect.get_image(kinect_frame=KinectFrames.DEPTH)
        image_processor = ImageProcessor(image=image)
        self.photos_taken[image_name][DEPTH_IMAGES_KEY] = image_processor

        # SAVE PROJECTOR IMAGE
        # image = self.projector_window.image
        # image_base = ImageBase(image=image)
        # self.photos_taken[image_name][PROJECTED_IMAGES_KEY] = image_base

        self.current_index = len(self.photos_taken) - 1
        self.update_image()

        self.enable_buttons()

    # endregion

    # region Patterns management

    def delete_image_information(self, image_name):
        if image_name in self.object_points.keys():
            self.object_points.pop(image_name)
        if image_name in self.image_points.keys():
            self.image_points.pop(image_name)
        if image_name in self.depth_points.keys():
            self.depth_points.pop(image_name)

    def delete_patterns(self):
        if messagebox.askokcancel("Borrar Patrones", "¿Estás seguro de borrar todos los patrones encontrados?"):
            for image_name, image_map in self.photos_taken.items():
                image_map[RGB_IMAGES_KEY].restore()
                self.delete_image_information(image_name=image_name)

            self.update_image()

    def calculate_patterns(self):
        # Check if we have the pattern size
        board_x_value = self.entry_size_x.get()
        board_y_value = self.entry_size_y.get()
        if not board_x_value or not board_y_value:
            messagebox.showerror("Missing argument", "Se necesita el tamaño del tablero para calcular los patrones.")
            return
        self.pattern_size = (int(board_x_value), int(board_y_value))

        # Stop all interactions with window
        self.disable_buttons()

        self.countdown_label.place(relx=0.5, rely=0.5, anchor="center")
        self.countdown_label.lift()
        self.countdown_label.config(text="Calculando Patrones...", font=("Helvetica", 25))

        self.window.update()

        # CALCULATE PATTERNS
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)

        image_names_to_delete = []
        for image_name, image_map in self.photos_taken.items():
            if image_name in self.object_points.keys():
                continue

            rgb_img = image_map[RGB_IMAGES_KEY]
            self.image_shape = rgb_img.image_shape
            depth_img = image_map[DEPTH_IMAGES_KEY]

            ret, corners = rgb_img.find_chessboard_corners(pattern_size=self.pattern_size)
            if ret:
                self.object_points[image_name] = pattern_points
                self.image_points[image_name] = corners

                # depth_points = []
                # for corner in corners:
                #     x, y = int(corner[0][0]), int(corner[0][1])
                #     x_transformed, y_transformed = transform_cords(point=(x, y), original_size=rgb_img.image_shape,
                #                                                    new_size=depth_img.image_shape)
                #     depth = depth_img.image[y_transformed, x_transformed]
                #     if depth == 0:
                #         continue
                #     depth_points.append([x, y, depth])
                # self.depth_points[image_name] = depth_points

                # TODO ADD SUBPIX???
                rgb_img.draw_chessboard_corners(pattern_size=self.pattern_size, corners=corners)
            else:
                # Delete image
                image_names_to_delete.append(image_name)

        if image_names_to_delete:
            for image_name in image_names_to_delete:
                self.photos_taken.pop(image_name)
            messagebox.showwarning("Fotos eliminadas",
                                   f"Se han eliminado {len(image_names_to_delete)} fotos dado que el patrón no se encuentra.")

        self.current_index = len(self.photos_taken) - 1

        # Activate buttons again
        self.enable_buttons()

        self.countdown_label.place_forget()

        self.update_image()

    # endregion
