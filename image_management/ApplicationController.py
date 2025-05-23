import threading

from literals import ConfigControllerEnum


class SharedConfig:
    def __init__(self):
        self.lock = threading.Lock()

        self.current_image = None
        self.second_image = None

        for config in ConfigControllerEnum:
            setattr(self, config.name, config.value)

    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_values(self):
        with self.lock:
            return {key.name: getattr(self, key.name) for key in ConfigControllerEnum}

    def get_value(self, key):
        with self.lock:
            return getattr(self, key)

    def set_value(self, key, value):
        with self.lock:
            setattr(self, key, value)
