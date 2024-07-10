import logging
import re
import tkinter as tk
from tkinter import ttk, messagebox

from screeninfo import screeninfo

from screen_controller.PrincipalScreenController import PrincipalScreenController
from screen_controller.ProjectorScreenController import ProjectorScreenController


def selector_screens():
    selector_window = tk.Tk()
    app = SelectorScreenInterface(window=selector_window)
    selector_window.mainloop()

    if app.process_interrupted:
        raise InterruptedError("Screen selection interrupted manually")

    return app.main_screen, app.projector_screen


class NumberScreenWindow:
    def __init__(self, window, position, window_name):
        self.window = window
        self.window.title(f"{window_name}")
        self.window.geometry(f"300x50+{position[0]}+{position[1]}")
        self.label = tk.Label(self.window, text=f"{window_name}", font=("Arial", 30))
        self.label.pack()


class SelectorScreenInterface:
    def __init__(self, window):
        self.window = window

        # Initiate screens main and projector
        self.main_screen = None
        self.projector_screen = None

        # Instantiate window
        self.window.title("Selección de Pantallas")

        self.frame = tk.Frame(self.window)
        self.frame.pack()

        self.screens = screeninfo.get_monitors()
        screen_names = [f"Screen: {re.sub(r'[^a-zA-Z0-9]', '', screen.name)}" for screen in self.screens]

        tk.Label(self.frame, text="Seleccione pantalla principal:").grid(row=0, column=0, padx=5, pady=5)
        self.main_screen_combobox = ttk.Combobox(self.frame, values=screen_names + ["Sin Pantalla Principal"])
        self.main_screen_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.main_screen_combobox.current(0)

        tk.Label(self.frame, text="Seleccione pantalla del proyector:").grid(row=1, column=0, padx=5, pady=5)
        self.projector_screen_combobox = ttk.Combobox(self.frame, values=screen_names)
        self.projector_screen_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.projector_screen_combobox.current(0 if len(screen_names) == 1 else 1)

        finish_button = tk.Button(self.frame, text="Aceptar", command=self.update_screens_after)
        finish_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.opened_windows = []
        for possible_screen in self.screens:
            window_screen = tk.Toplevel()
            new_window = NumberScreenWindow(window=window_screen, position=(possible_screen.x, possible_screen.y),
                                            window_name=re.sub(r'[^a-zA-Z0-9]', '', possible_screen.name))
            self.opened_windows.append(new_window)

        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry('{}x{}+{}+{}'.format(width, height, x + width // 2, y + height // 2))

        # Generate code for exit closing the window
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_interrupted = False

    def on_closing(self):
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la ventana? El proceso terminará."):
            self.process_interrupted = True
            self.destroy()

    def destroy(self):
        for active_window in self.opened_windows:
            active_window.window.destroy()
        self.window.destroy()

    def update_screens_after(self):
        self.window.after(0, self.update_screens())

    def update_screens(self):
        main_screen_index = self.main_screen_combobox.current()
        projector_screen_index = self.projector_screen_combobox.current()

        if main_screen_index == projector_screen_index:
            logging.warning(
                "Cannot use same screen for projector and principal, principal screen will be changed to 'No Pantalla Principal' value")
        else:
            # Principal screen
            main_screen_tmp = self.screens[main_screen_index]
            self.main_screen = PrincipalScreenController(
                screen_name=re.sub(r'[^a-zA-Z0-9]', '', main_screen_tmp.name),
                width_resolution=main_screen_tmp.width,
                height_resolution=main_screen_tmp.height,
                position=(main_screen_tmp.x, main_screen_tmp.y))

        # Projector Screen
        projector_screen_tmp = self.screens[projector_screen_index]
        self.projector_screen = ProjectorScreenController(
            screen_name=re.sub(r'[^a-zA-Z0-9]', '', projector_screen_tmp.name),
            width_resolution=projector_screen_tmp.width,
            height_resolution=projector_screen_tmp.height,
            position=(projector_screen_tmp.x, projector_screen_tmp.y))

        self.destroy()
