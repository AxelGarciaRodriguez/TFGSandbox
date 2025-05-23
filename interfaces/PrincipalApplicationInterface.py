import tkinter as tk
from tkinter import ttk, messagebox
import cv2

from PIL import Image, ImageTk

from image_management.ApplicationController import SharedConfig
from literals import ConfigControllerNamesEnum, ConfigControllerSliderEnum, ConfigControllerEnum


def instantiate_principal_application_interface(config, principal_screen):
    root = tk.Tk()
    root.geometry(f"+{principal_screen.position[0]}+{principal_screen.position[1]}")
    app = PrincipalApplicationInterface(window=root, config=config)
    root.mainloop()

    return app


class PrincipalApplicationInterface:

    def __init__(self, window, config):
        self.config = config

        self.root = window
        self.root.title("Configuración")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.grid(row=0, column=0, sticky="n", padx=10, pady=10)

        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        self.colormaps = [attr for attr in dir(cv2) if attr.startswith("COLORMAP_")]
        self.colormap_values = {name: getattr(cv2, name) for name in self.colormaps}

        self.entries = {}
        for config_name in ConfigControllerNamesEnum:
            if config_name.name == ConfigControllerNamesEnum.COLORMAP.name:
                self.entries[config_name.name] = self.add_dropdown(self.controls_frame, config_name.value,
                                                                   config_name.name,
                                                                   values=list(self.colormap_values.keys()))

            elif config_name.name in ConfigControllerSliderEnum.__members__.keys():
                slider_control = getattr(ConfigControllerSliderEnum, config_name.name)
                self.entries[config_name.name] = self.add_slider(self.controls_frame, config_name.value,
                                                                 config_name.name,
                                                                 from_=slider_control.value[0],
                                                                 to=slider_control.value[1])
            else:
                self.entries[config_name.name] = self.add_entry(self.controls_frame, config_name.value,
                                                                config_name.name)

        tk.Button(self.controls_frame, text="Reiniciar Imagen", command=self.reset_image).pack(pady=5)

        tk.Button(self.controls_frame, text="Actualizar configuración", command=self.update_config).pack(pady=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, pady=5)

        self.second_image_label = tk.Label(self.image_frame)
        self.second_image_label.grid(row=1, column=0, pady=5)

        self.update_image()

    def add_entry(self, frame, label_text, varname):
        tk.Label(frame, text=label_text).pack(anchor="w")
        entry = tk.Entry(frame)
        entry.insert(0, str(getattr(self.config, varname)))
        entry.pack(fill='x')
        return entry

    def add_slider(self, frame, label_text, varname, from_, to):
        tk.Label(frame, text=label_text).pack(anchor="w")
        val = getattr(self.config, varname)
        scale = tk.Scale(frame, from_=from_, to=to, orient="horizontal")
        scale.set(val)
        scale.pack(fill='x')
        return scale

    def add_dropdown(self, frame, label_text, varname, values):
        tk.Label(frame, text=label_text).pack(anchor="w")
        val = [k for k, v in self.colormap_values.items() if v == getattr(self.config, varname)][0]
        var = tk.StringVar(value=val)
        dropdown = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
        dropdown.pack(fill='x')
        return var

    def reset_image(self):
        self.config.update(RESET_IMAGE=True)
        messagebox.showinfo("Imagen", "Flag de reset de imagen activado.")

    def update_config(self):
        try:
            values = {}
            for key in self.entries.keys():
                if key == ConfigControllerEnum.COLORMAP.name:
                    values[key] = self.colormap_values[self.entries[key].get()]
                else:
                    values[key] = int(self.entries[key].get())

            # CONTROL VARIABLE
            if not values[ConfigControllerNamesEnum.MIN_DEPTH.name] > 0:
                raise ValueError(f"{ConfigControllerNamesEnum.MIN_DEPTH.value} debe ser > 0")
            if not values[ConfigControllerNamesEnum.MAX_DEPTH.name] > 0:
                raise ValueError(f"{ConfigControllerNamesEnum.MAX_DEPTH.value} debe ser > 0")
            if not values[ConfigControllerNamesEnum.MIN_DEPTH.name] < values[
                ConfigControllerNamesEnum.MAX_DEPTH.name]:
                raise ValueError(
                    f"{ConfigControllerNamesEnum.MIN_DEPTH.value} debe ser < {ConfigControllerNamesEnum.MAX_DEPTH.value}")

            if not values[ConfigControllerNamesEnum.ERRORS_UMBRAL.name] <= values[
                ConfigControllerNamesEnum.MEDIUM_NOISE.name]:
                raise ValueError(
                    f"{ConfigControllerNamesEnum.ERRORS_UMBRAL.value} debe ser <= {ConfigControllerNamesEnum.MEDIUM_NOISE.value}")
            if not values[ConfigControllerNamesEnum.MEDIUM_NOISE.name] <= values[
                ConfigControllerNamesEnum.BIG_NOISE.name]:
                raise ValueError(
                    f"{ConfigControllerNamesEnum.MEDIUM_NOISE.value} debe ser <= {ConfigControllerNamesEnum.BIG_NOISE.value}")

            self.config.update(**values)
            messagebox.showinfo("Éxito", "Configuración actualizada")

        except ValueError as e:
            messagebox.showerror("Error de validación", str(e))

    def update_image(self):
        with self.config.lock:
            frame = self.config.current_image
            second = self.config.second_image

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((320, 240))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img
        else:
            pil_img = Image.new("RGB", (320, 240), color=(80, 80, 80))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img

        if second is not None:
            rgb2 = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
            pil_img2 = Image.fromarray(rgb2).resize((320, 240))
            tk_img2 = ImageTk.PhotoImage(pil_img2)
            self.second_image_label.config(image=tk_img2)
            self.second_image_label.image = tk_img2
        else:
            pil_img2 = Image.new("RGB", (320, 240), color=(120, 120, 120))
            tk_img2 = ImageTk.PhotoImage(pil_img2)
            self.second_image_label.config(image=tk_img2)
            self.second_image_label.image = tk_img2

        self.root.after(100, self.update_image)
