def generate_cords(width, height, initial_position=None):
    if not initial_position:
        initial_position = (0, 0)

    return [initial_position, (initial_position[0], initial_position[1] + height),
            (initial_position[0] + width, initial_position[1]),
            (initial_position[0] + width, initial_position[1] + height)]


