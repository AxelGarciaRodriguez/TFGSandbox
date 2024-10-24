import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


def instantiate_draw_polygon_interface(image, depth_image, principal_screen, window_size=(1920, 1080)):
    calibrate_window = tk.Tk()
    calibrate_window.geometry(
        f"{window_size[0]+30}x{window_size[1]+30}+{principal_screen.position[0]}+{principal_screen.position[1]}")
    app = DrawPolygonInterface(window=calibrate_window, image=image,
                               depth_image=depth_image, window_size=window_size)
    calibrate_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Calibration interrupted manually")

    return app


class DrawPolygonInterface:
    def __init__(self, window, image, depth_image, window_size=(1920, 1080)):
        # Necessary variables
        self.image = image
        self.depth_image = depth_image
        self.window_size = window_size

        image_shape = image.shape[:2]
        self.initial_height, self.initial_width = image_shape
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

            self.points.append((x_image, y_image))
            self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="red")
            if self.depth_image is not None:
                z = self.depth_image[int(y_image), int(x_image)]
                self.canvas.create_text(event.x, event.y - 10, text=f"({int(x_image)}, {int(y_image)}, {z})", fill="black",
                                        font='Helvetica 10 bold')
            else:
                self.canvas.create_text(event.x, event.y - 10, text=f"({int(x_image)}, {int(y_image)})", fill="black",
                                        font='Helvetica 10 bold')
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
        self.image = Image.fromarray(self.image)
        self.image = self.image.resize(self.window_size)
        self.photo = ImageTk.PhotoImage(self.image)
        self.photo_obj = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def accept_polygon(self):
        self.window.destroy()

    def accept_polygon_after(self):
        self.window.after(0, self.accept_polygon())
