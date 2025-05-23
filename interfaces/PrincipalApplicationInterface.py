import tkinter as tk
from tkinter import ttk, messagebox
import cv2

from PIL import Image, ImageTk

from image_management.ApplicationController import SharedConfig
from literals import ConfigControllerNamesEnum, ConfigControllerSliderEnum, ConfigControllerEnum


def start_gui(config: SharedConfig):
    root = tk.Tk()
    root.title("Configuración")

    main_frame = tk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    controls_frame = tk.Frame(main_frame)
    controls_frame.grid(row=0, column=0, sticky="n", padx=10, pady=10)

    image_frame = tk.Frame(main_frame)
    image_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

    colormaps = [attr for attr in dir(cv2) if attr.startswith("COLORMAP_")]
    colormap_values = {name: getattr(cv2, name) for name in colormaps}

    def add_entry(frame, label_text, varname):
        tk.Label(frame, text=label_text).pack(anchor="w")
        entry = tk.Entry(frame)
        entry.insert(0, str(getattr(config, varname)))
        entry.pack(fill='x')
        return entry

    def add_slider(frame, label_text, varname, from_, to):
        tk.Label(frame, text=label_text).pack(anchor="w")
        val = getattr(config, varname)
        scale = tk.Scale(frame, from_=from_, to=to, orient="horizontal")
        scale.set(val)
        scale.pack(fill='x')
        return scale

    def add_dropdown(frame, label_text, varname, values):
        tk.Label(frame, text=label_text).pack(anchor="w")
        val = [k for k, v in colormap_values.items() if v == getattr(config, varname)][0]
        var = tk.StringVar(value=val)
        dropdown = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
        dropdown.pack(fill='x')
        return var

    def reset_image():
        config.update(RESET_IMAGE=True)
        messagebox.showinfo("Imagen", "Flag de reset de imagen activado.")

    entries = {}
    for config_name in ConfigControllerNamesEnum:
        if config_name.name == ConfigControllerNamesEnum.COLORMAP.name:
            entries[config_name.name] = add_dropdown(controls_frame, config_name.value, config_name.name,
                                                     values=list(colormap_values.keys()))

        elif config_name.name in ConfigControllerSliderEnum.__members__.keys():
            slider_control = getattr(ConfigControllerSliderEnum, config_name.name)
            entries[config_name.name] = add_slider(controls_frame, config_name.value, config_name.name,
                                                   from_=slider_control.value[0], to=slider_control.value[1])
        else:
            entries[config_name.name] = add_entry(controls_frame, config_name.value, config_name.name)

    tk.Button(controls_frame, text="Reiniciar Imagen", command=reset_image).pack(pady=5)

    def update_config():
        try:
            values = {}
            for key in entries.keys():
                if key == ConfigControllerEnum.COLORMAP.name:
                    values[key] = colormap_values[entries[key].get()]
                else:
                    values[key] = int(entries[key].get())

            # CONTROL VARIABLE
            if not values[ConfigControllerNamesEnum.MIN_DEPTH.name] > 0:
                raise ValueError(f"{ConfigControllerNamesEnum.MIN_DEPTH.value} debe ser > 0")
            if not values[ConfigControllerNamesEnum.MAX_DEPTH.name] > 0:
                raise ValueError(f"{ConfigControllerNamesEnum.MAX_DEPTH.value} debe ser > 0")
            if not values[ConfigControllerNamesEnum.MIN_DEPTH.name] < values[ConfigControllerNamesEnum.MAX_DEPTH.name]:
                raise ValueError(f"{ConfigControllerNamesEnum.MIN_DEPTH.value} debe ser < {values[ConfigControllerNamesEnum.MAX_DEPTH.value]}")

            config.update(**values)
            messagebox.showinfo("Éxito", "Configuración actualizada")

        except ValueError as e:
            messagebox.showerror("Error de validación", str(e))

    tk.Button(controls_frame, text="Actualizar configuración", command=update_config).pack(pady=10)

    image_label = tk.Label(image_frame)
    image_label.grid(row=0, column=0, pady=5)

    second_image_label = tk.Label(image_frame)
    second_image_label.grid(row=1, column=0, pady=5)

    def update_image():
        with config.lock:
            frame = config.current_image
            second = config.second_image

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((320, 240))
            tk_img = ImageTk.PhotoImage(pil_img)
            image_label.config(image=tk_img)
            image_label.image = tk_img
        else:
            pil_img = Image.new("RGB", (320, 240), color=(80, 80, 80))
            tk_img = ImageTk.PhotoImage(pil_img)
            image_label.config(image=tk_img)
            image_label.image = tk_img

        if second is not None:
            rgb2 = cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
            pil_img2 = Image.fromarray(rgb2).resize((320, 240))
            tk_img2 = ImageTk.PhotoImage(pil_img2)
            second_image_label.config(image=tk_img2)
            second_image_label.image = tk_img2
        else:
            pil_img2 = Image.new("RGB", (320, 240), color=(120, 120, 120))
            tk_img2 = ImageTk.PhotoImage(pil_img2)
            second_image_label.config(image=tk_img2)
            second_image_label.image = tk_img2

        root.after(100, update_image)

    update_image()

    root.mainloop()
