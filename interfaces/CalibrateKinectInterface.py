import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cv2
import numpy as np

from image_manager.ImageProcessorDepth import ImageProcessorDepth
from image_manager.ImageProcessorIR import ImageProcessorIR
from image_manager.ImageProcessorRGB import ImageProcessorRGB
from literals import OBJ_POINTS_KEY, IMG_POINTS_KEY, CALIBRATE_PATTERN_IMAGES, CAMERA_CALIBRATION_VARIABLE, \
    CAMERA_DISTORTION_VARIABLE, CAMERA_ROTATION_VARIABLE, \
    CAMERA_TRANSLATION_VARIABLE, KinectFrames

from PIL import Image, ImageTk

from literals_control import PATTERN_SIZE, PATTERN_METERS


def instantiate_calibrate_kinect_interface(kinect, previous_images, not_found_image, principal_screen,
                                           window_image_size=(1920, 1080)):
    calibrate_window = tk.Tk()
    calibrate_window.geometry(
        f"{window_image_size[0] + 30}x{window_image_size[1] + 150}+{principal_screen.position[0]}+{principal_screen.position[1]}")
    app = CalibrateKinectInterface(window=calibrate_window, kinect=kinect, previous_images=previous_images,
                                   not_found_image=not_found_image, window_image_size=window_image_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class CalibrateKinectInterface:

    def __init__(self, window, kinect, previous_images, not_found_image, window_image_size=(960, 540)):
        # Necessary variables
        self.kinect = kinect
        self.pattern_size = PATTERN_SIZE
        self.board_size = PATTERN_METERS
        self.photos_taken = previous_images  # Previous images contains 1 key for each image, and a dict for COLOR, depth and pattern images

        self.window_image_size = window_image_size
        self.not_found_image = not_found_image

        self.image_information = {}  # Map with image name as key
        self.camera_information = {}  # Map with camera name as key

        # Initiate window
        self.window = window
        self.window.title("Calibrate Kinect Cameras")

        self.frame = tk.Frame(self.window)
        self.frame.pack()

        row_counter = 0

        # Generate window board size registers
        tk.Label(self.frame, text="Tamaño de la matriz (ancho x alto):").grid(row=row_counter, column=0, columnspan=2,
                                                                              padx=5,
                                                                              pady=5)
        self.entry_size_x = tk.Entry(self.frame)
        self.entry_size_x.grid(row=row_counter, column=2)
        self.entry_size_x.insert(0, f"{self.pattern_size[0]}")

        tk.Label(self.frame, text="X").grid(row=0, column=3)

        self.entry_size_y = tk.Entry(self.frame)
        self.entry_size_y.grid(row=row_counter, column=4)
        self.entry_size_y.insert(0, f"{self.pattern_size[1]}")

        row_counter += 1
        tk.Label(self.frame, text="Tamaño del tablero en m (cada fila/columna):").grid(row=row_counter, column=0,
                                                                                       columnspan=2,
                                                                                       padx=5, pady=5)
        self.entry_size_z = tk.Entry(self.frame)
        self.entry_size_z.grid(row=1, column=2)
        self.entry_size_z.insert(0, f"{self.board_size}")

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

        self.button_exit = tk.Button(self.frame, text="Terminar Calibración", command=self.calibrate_cameras)
        self.button_exit.grid(row=row_counter, column=4, padx=5, pady=5)

        self.update_image()

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
        if len(self.photos_taken) != len(self.image_information):
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

        image_processor = list(self.photos_taken.values())[self.current_index][KinectFrames.COLOR.name]
        image = Image.fromarray(image_processor.image[..., ::-1])
        image = image.resize(self.window_image_size)
        photo = ImageTk.PhotoImage(image)
        self.last_image.config(image=photo)
        self.last_image.image = photo

        self.index_label.config(text="{} / {}".format(self.current_index + 1, len(self.photos_taken)))
        self.prev_button.config(state=tk.NORMAL)
        self.button_delete_photo.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)

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
                self.delete_image_name(image_name=image_name)

                self.current_index = self.current_index - 1 if self.current_index > 0 else 0
                self.update_image()

    def delete_image_name(self, image_name):
        if self.photos_taken and image_name in self.photos_taken.keys():
            self.photos_taken.pop(image_name)
            self.delete_image_information(image_name=image_name)

    def take_image(self):
        self.disable_buttons()

        self.countdown_label.place(relx=0.5, rely=0.5, anchor="center")
        self.countdown_label.lift()
        for i in range(3, 0, -1):
            self.countdown(i)
            self.window.update()
            self.window.after(1000)

        self.countdown_label.place_forget()

        image_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'

        self.photos_taken[image_name] = {}
        # GET RGB IMAGE
        image = self.kinect.get_image(kinect_frame=KinectFrames.COLOR)
        image_processor = ImageProcessorRGB(image=image)
        self.photos_taken[image_name][KinectFrames.COLOR.name] = image_processor

        # GET IR IMAGE
        image = self.kinect.get_image(kinect_frame=KinectFrames.INFRARED)
        image_processor = ImageProcessorIR(image=image)
        self.photos_taken[image_name][KinectFrames.INFRARED.name] = image_processor

        # GET DEPTH IMAGE
        # image = self.kinect.get_image(kinect_frame=KinectFrames.DEPTH)
        # image_processor = ImageProcessorDepth(image=image)
        # self.photos_taken[image_name][KinectFrames.DEPTH.name] = image_processor

        self.current_index = len(self.photos_taken) - 1
        self.update_image()

        self.enable_buttons()

    # endregion

    # region Patterns management

    def delete_image_information(self, image_name):
        if image_name in self.image_information.keys():
            self.image_information.pop(image_name)

    def delete_patterns(self):
        if messagebox.askokcancel("Borrar Patrones", "¿Estás seguro de borrar todos los patrones encontrados?"):
            for image_name, image_map in self.photos_taken.items():
                image_map[KinectFrames.COLOR.name].restore()
                image_map[KinectFrames.INFRARED.name].restore()
                self.delete_image_information(image_name=image_name)

            self.update_image()

    def calibrate_cameras(self):
        self.calculate_patterns()

        # CALIBRATE CAMERAS RGB AND IR
        obj_points = {}
        img_points = {}
        example_img = {}
        for calibrate_cameras in CALIBRATE_PATTERN_IMAGES:
            obj_points[calibrate_cameras] = []
            img_points[calibrate_cameras] = []
            example_img[calibrate_cameras] = None
            for image_name, image_map in self.image_information.items():
                obj_points[calibrate_cameras].append(image_map[calibrate_cameras][OBJ_POINTS_KEY])
                img_points[calibrate_cameras].append(image_map[calibrate_cameras][IMG_POINTS_KEY])
                example_img[calibrate_cameras] = self.photos_taken[image_name][calibrate_cameras]

            rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points[calibrate_cameras],
                                                                               img_points[calibrate_cameras],
                                                                               (example_img[calibrate_cameras].width,
                                                                                example_img[calibrate_cameras].height),
                                                                               None,
                                                                               None,
                                                                               criteria=(
                                                                                   cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                                                   120, 0.001),
                                                                               flags=0)

            self.camera_information[calibrate_cameras] = {
                CAMERA_CALIBRATION_VARIABLE: camera_matrix,
                CAMERA_DISTORTION_VARIABLE: dist_coefs,
                CAMERA_ROTATION_VARIABLE: rvecs,
                CAMERA_TRANSLATION_VARIABLE: tvecs,
                OBJ_POINTS_KEY: obj_points[calibrate_cameras],
                IMG_POINTS_KEY: img_points[calibrate_cameras]
            }

        # CALIBRATE STEREO
        pattern_points = np.zeros((len(self.photos_taken.keys()), np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= self.board_size

        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(pattern_points,
                                                                                                         img_points[
                                                                                                             KinectFrames.INFRARED.name],
                                                                                                         img_points[
                                                                                                             KinectFrames.COLOR.name],
                                                                                                         self.camera_information[
                                                                                                             KinectFrames.INFRARED.name][
                                                                                                             CAMERA_CALIBRATION_VARIABLE],
                                                                                                         self.camera_information[
                                                                                                             KinectFrames.INFRARED.name][
                                                                                                             CAMERA_DISTORTION_VARIABLE],
                                                                                                         self.camera_information[
                                                                                                             KinectFrames.COLOR.name][
                                                                                                             CAMERA_CALIBRATION_VARIABLE],
                                                                                                         self.camera_information[
                                                                                                             KinectFrames.COLOR.name][
                                                                                                             CAMERA_DISTORTION_VARIABLE],
                                                                                                         (example_img[
                                                                                                              KinectFrames.INFRARED.name].width,
                                                                                                          example_img[
                                                                                                              KinectFrames.INFRARED.name].height))
        # TODO MANAGE CALIBRATE STEREO INFORMATION ???
        self.exit_after()

    def calculate_patterns(self):
        # Check if we have the pattern size
        board_x_value = self.entry_size_x.get()
        board_y_value = self.entry_size_y.get()
        if not board_x_value or not board_y_value:
            messagebox.showerror("Missing argument", "Se necesita el tamaño del tablero para calcular los patrones.")
            return
        self.pattern_size = (int(board_x_value), int(board_y_value))

        board_size_value = self.entry_size_z.get()
        if not board_size_value:
            messagebox.showerror("Missing argument", "Se necesita el tamaño del cuadrado para calcular los patrones.")
            return
        self.board_size = float(board_size_value)

        # Stop all interactions with window
        self.disable_buttons()

        self.countdown_label.place(relx=0.5, rely=0.5, anchor="center")
        self.countdown_label.lift()
        self.countdown_label.config(text="Calculando Patrones...", font=("Helvetica", 25))

        self.window.update()

        # SEARCH PATTERN CHESS
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= self.board_size

        image_names_to_delete = []
        total_images_used = len(self.photos_taken.keys())
        for image_name, image_map in self.photos_taken.items():
            if image_name in self.image_information.keys():
                continue

            corners_map = {}
            for type_image in CALIBRATE_PATTERN_IMAGES:
                image_processor = image_map[type_image]
                ret, corners = image_processor.find_chessboard_corners(pattern_size=self.pattern_size)
                if ret:
                    corners_map[type_image] = image_processor.calculate_sub_pix_corner(corners=corners)
                    image_processor.draw_chessboard_corners(pattern_size=self.pattern_size, corners=corners)
                else:
                    image_names_to_delete.append(image_name)
                    total_images_used -= 1
                    break
            else:
                self.image_information[image_name] = {}
                for type_image in CALIBRATE_PATTERN_IMAGES:
                    self.image_information[image_name][type_image] = {
                        OBJ_POINTS_KEY: pattern_points,
                        IMG_POINTS_KEY: corners_map[type_image].reshape(-1, 2)
                    }

        ######################### TEST CENTRAL POINT RGB
        # rgb_image = self.kinect.get_image(kinect_frame=KinectFrames.COLOR)
        # infrared_image = self.kinect.get_image(kinect_frame=KinectFrames.INFRARED)
        #
        # center_rgb = (rgb_image.shape[1] // 2, rgb_image.shape[0] // 2)
        # infrared_point = transform_point(point=center_rgb, R=R, T=T,
        #                                  mtx_rgb=self.camera_information[KinectFrames.COLOR.name][CAMERA_CALIBRATION_VARIABLE],
        #                                  dist_rgb=self.camera_information[KinectFrames.COLOR.name][CAMERA_DISTORTION_VARIABLE],
        #                                  mtx_ir=self.camera_information[KinectFrames.INFRARED.name][CAMERA_CALIBRATION_VARIABLE],
        #                                  dist_ir=self.camera_information[KinectFrames.INFRARED.name][CAMERA_DISTORTION_VARIABLE])
        #
        # rgb_image_processor = ImageProcessorRGB(image=rgb_image)
        # infrared_image_processor = ImageProcessorIR(image=infrared_image)
        #
        # rgb_image_processor.draw_point(point=center_rgb)
        # infrared_image_processor.draw_point(point=infrared_point)
        #
        # cv2.imshow('Left Rectified', rgb_image_processor.image)
        # cv2.imshow('Right Rectified', infrared_image_processor.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # ######################## TEST RECTIFY IMAGES
        # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (
        #     example_img[KinectFrames.INFRARED.name].width, example_img[KinectFrames.INFRARED.name].height), R, T)
        #
        # left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (
        #     example_img[KinectFrames.COLOR.name].width, example_img[KinectFrames.COLOR.name].height), cv2.CV_16SC2)
        # right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (
        #     example_img[KinectFrames.INFRARED.name].width, example_img[KinectFrames.INFRARED.name].height), cv2.CV_16SC2)
        #
        # rgb_image = self.kinect.get_image(kinect_frame=KinectFrames.COLOR)
        # infrared_image = self.kinect.get_image(kinect_frame=KinectFrames.INFRARED)
        #
        # left_rectified = cv2.remap(rgb_image, left_map1, left_map2, cv2.INTER_LINEAR)
        # right_rectified = cv2.remap(infrared_image, right_map1, right_map2, cv2.INTER_LINEAR)
        #
        # cv2.imshow('Left Rectified', left_rectified)
        # cv2.imshow('Right Rectified', right_rectified)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # ###############################################

        # DELETE UNNECESSARY IMAGES
        if image_names_to_delete:
            for image_name in image_names_to_delete:
                self.delete_image_name(image_name=image_name)

            messagebox.showwarning("Fotos eliminadas",
                                   f"Se han eliminado {len(image_names_to_delete)} fotos dado que el patrón no se encuentra.")

        self.current_index = len(self.photos_taken) - 1

        # Activate buttons again
        self.enable_buttons()

        self.countdown_label.place_forget()

        self.update_image()

    # endregion


def transform_point(point, R, T, mtx_rgb, dist_rgb, mtx_ir, dist_ir):
    # Convertir el punto a coordenadas homogéneas
    point_homogeneous = np.array([point[0], point[1], 1.0])

    # Calibrar el punto de la imagen RGB
    undistorted_point = cv2.undistortPoints(np.array([[point_homogeneous]], dtype=np.float32), mtx_rgb, dist_rgb)

    # Convertir a coordenadas de la cámara RGB
    point_camera_rgb = np.dot(np.linalg.inv(mtx_rgb), np.append(undistorted_point[0][0], 1.0))

    # Transformar el punto a la cámara IR
    point_camera_ir = np.dot(R, point_camera_rgb[:3]) + T

    # Convertir a coordenadas de la imagen IR
    point_camera_ir = point_camera_ir.reshape(-1, 1, 3)
    point_image_ir, _ = cv2.projectPoints(point_camera_ir, np.zeros((3, 1)), np.zeros((3, 1)), mtx_ir, dist_ir)

    return point_image_ir[0][0]
